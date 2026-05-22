from __future__ import annotations

import os

# 禁止 timm/huggingface 在线下载（推理阶段完全不需要，避免网络超时）
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import torch


def _add_src_to_path() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    src = repo_root / "src"
    sys.path.insert(0, str(src))


def _setup_logger(experiment_dir: Path) -> logging.Logger:
    logger = logging.getLogger("anomalib_experiment")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter("[%(asctime)s] %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    logger.addHandler(console)

    log_file = experiment_dir / "predict.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger


def _find_ckpt(train_dir: Path) -> Path:
    """在训练目录下自动查找 checkpoint。"""
    # 1) 精确匹配 last.ckpt
    ckpt = train_dir / "weights" / "lightning" / "last.ckpt"
    if ckpt.is_file():
        return ckpt

    # 2) 在 weights/lightning 下找任意 .ckpt
    candidates = sorted((train_dir / "weights" / "lightning").glob("*.ckpt"))
    if candidates:
        return candidates[-1]

    # 3) 整个训练目录递归找 .ckpt（兜底）
    all_ckpts = sorted(train_dir.rglob("*.ckpt"))
    if all_ckpts:
        return all_ckpts[-1]

    raise FileNotFoundError(
        f"No checkpoint found under: {train_dir}\n"
        f"Expected: {train_dir / 'weights' / 'lightning' / 'last.ckpt'}"
    )


def _timm_layer_to_idx(backbone: str, layers: list[str]) -> list[int]:
    """将 layer 名称映射为 timm 模型的输出索引（与 TimmFeatureExtractor 逻辑一致）。"""
    import timm

    model = timm.create_model(
        backbone,
        pretrained=False,
        features_only=True,
        exportable=True,
    )
    layer_names = [info["module"] for info in model.feature_info.info]
    idx = []
    for layer in layers:
        try:
            idx.append(layer_names.index(layer))
        except ValueError:
            raise ValueError(
                f"Layer {layer} not found in model {backbone}. Available: {layer_names}"
            )
    del model
    return idx


def _load_backbone_weights(args: argparse.Namespace, log: logging.Logger):
    """
    加载预训练权重，返回 backbone_arg (模型实例 / 字符串)。
    必须与 train.py 的逻辑保持一致，否则 checkpoint 的 state_dict 键名对不上！
    """
    backbone_arg = args.backbone

    if args.model == "simplenet":
        log.info(f"  预训练权重: SimpleNet 使用 timm .tv 格式 ({args.backbone})")
        return backbone_arg

    if args.mmdet_weights:
        weights_path = Path(args.mmdet_weights)
        if not weights_path.is_file():
            log.error(f"❌ 找不到 MMDetection 权重文件: {weights_path.resolve()}")
            sys.exit(1)
        log.info(f"  正在加载 MMDet 权重 : {weights_path.resolve()}")
        import torchvision.models as tv_models
        if hasattr(tv_models, args.backbone):
            backbone_arg = getattr(tv_models, args.backbone)()
        else:
            log.error(f"❌ torchvision 中不支持该模型: {args.backbone}")
            sys.exit(1)
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        bb_state = {k.replace("backbone.", ""): v
                    for k, v in state_dict.items() if k.startswith("backbone.")}
        if len(bb_state) == 0:
            bb_state = state_dict
        backbone_arg.load_state_dict(bb_state, strict=False)
        log.info(f"  ✅ MMDet权重灌入成功，匹配 {len(bb_state)} 个 Tensor")
        return backbone_arg

    if args.weights_path:
        weights_path = Path(args.weights_path)
        if weights_path.is_file():
            log.info(f"  加载本地预训练权重: {weights_path.resolve()} (与训练保持一致)")
            import timm
            bb_model = timm.create_model(
                args.backbone, pretrained=False,
                features_only=True, exportable=True,
                out_indices=_timm_layer_to_idx(args.backbone, args.layers),
            )
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
            bb_model.load_state_dict(state_dict, strict=False)
            log.info(f"  ✅ 本地权重加载成功 → backbone 为 nn.Module 实例")
            return bb_model
        else:
            log.warning(f"  ⚠ --weights-path 指定的文件不存在: {weights_path}")

    # 默认：字符串形式，由模型内部调 timm 在线下载
    if args.pre_trained:
        log.info("  预训练权重: 字符串模式，模型内部从 timm 加载")
    else:
        log.info("  预训练权重: 不使用 (随机初始化)")
    return backbone_arg


def main(args: argparse.Namespace | None = None) -> None:
    _add_src_to_path()

    from anomalib.engine import Engine
    from anomalib.models.image.patchcore.lightning_model import Patchcore
    from anomalib.models.image.supersimplenet.lightning_model import Supersimplenet
    from anomalib.models.image.reverse_distillation.lightning_model import ReverseDistillation
    from anomalib.models.image.padim.lightning_model import Padim
    from anomalib.post_processing import PostProcessor
    from anomalib.visualization.image.visualizer import ImageVisualizer

    args = _parse_args() if args is None else args

    # ---- 输出目录：predict_<timestamp> ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_root = Path("results") / f"predict_{timestamp}"
    images_dir = results_root / "images"
    results_root.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    log = _setup_logger(results_root)
    log.info("=" * 60)
    log.info(f"PREDICT  output: {results_root.resolve()}")
    log.info("=" * 60)

    # ---- 训练目录 & Checkpoint ----
    train_dir = Path(args.train_dir)
    ckpt_path = _find_ckpt(train_dir)
    log.info(f"train_dir = {train_dir}")
    log.info(f"ckpt_path = {ckpt_path}")
    log.info(f"model     = {args.model}")
    log.info(f"backbone  = {args.backbone}")

    # ---- 预测数据 ----
    data_path = Path(args.predict_path)
    log.info(f"data_path = {data_path}")

    # ---- 模型 & 后处理 ----
    post_processor = PostProcessor(
        image_sensitivity=args.image_sensitivity,
        pixel_sensitivity=args.pixel_sensitivity,
    )
    log.info(f"pixel_threshold = {1.0 - args.pixel_sensitivity:.2f} (sensitivity={args.pixel_sensitivity})")

    # ---- 加载 backbone（必须与 train.py 逻辑完全一致）----
    backbone_arg = _load_backbone_weights(args, log)

    use_pretrained = (args.weights_path is None and not args.mmdet_weights) and args.pre_trained
    common_kwargs = dict(
        visualizer=ImageVisualizer(output_dir=images_dir),
        post_processor=post_processor,
    )

    if args.model == "patchcore":
        model = Patchcore(
            backbone=backbone_arg,
            layers=tuple(args.layers),
            pre_trained=use_pretrained,
            coreset_sampling_ratio=args.coreset_sampling_ratio,
            **common_kwargs,
        )
    elif args.model == "simplenet":
        model = Supersimplenet(
            backbone=backbone_arg,
            layers=args.layers,
            perlin_threshold=args.perlin_threshold,
            **common_kwargs,
        )
    elif args.model == "rd":
        model = ReverseDistillation(
            backbone=backbone_arg,
            layers=tuple(args.layers),
            pre_trained=use_pretrained,
            input_size=tuple(args.input_size),
            anomaly_map_mode=args.anomaly_map_mode,
            **common_kwargs,
        )
    elif args.model == "padim":
        model = Padim(
            backbone=backbone_arg,
            layers=args.layers,
            pre_trained=use_pretrained,
            n_features=args.n_features,
            **common_kwargs,
        )
    else:
        log.error(f"❌ 不支持的模型: {args.model} (可选: patchcore/simplenet/rd/padim)")
        sys.exit(1)

    engine = Engine(
        default_root_dir=results_root,
        accelerator=args.accelerator,
        devices=args.devices,
    )

    t0 = time.time()
    log.info("Starting prediction...")
    predictions = engine.predict(
        model=model, ckpt_path=ckpt_path, data_path=data_path, return_predictions=True
    )
    elapsed = time.time() - t0
    log.info(f"Prediction finished in {elapsed:.1f}s")

    log.info(f"predictions_count: {len(predictions) if hasattr(predictions, '__len__') else '?'}")
    log.info(f"images saved to: {images_dir}")
    log.info("=" * 60)


# ================================================================
# 模型默认值（与 train.py 保持一致）
# ================================================================
_MODEL_DEFAULTS = {
    "patchcore": {"backbone": "resnet18", "layers": ["layer2", "layer3"]},
    "simplenet":  {"backbone": "resnet18.tv_in1k", "layers": ["layer2", "layer3"]},
    "rd":         {"backbone": "resnet18", "layers": ["layer1", "layer2", "layer3"]},
    "padim":      {"backbone": "resnet18", "layers": ["layer2", "layer3"], "n_features": 100},
}


def _apply_model_defaults(args: argparse.Namespace) -> None:
    """根据 --model 覆盖 backbone/layers/n_features 默认值。"""
    cfg = _MODEL_DEFAULTS[args.model]
    if args.backbone == "resnet18" and args.model in ("simplenet",):
        args.backbone = cfg["backbone"]
    if args.backbone == "wide_resnet50_2":
        args.backbone = cfg["backbone"]
    if args.layers == ["layer2", "layer3"] and args.model in ("rd",):
        args.layers = cfg["layers"]
    # PaDiM + nn.Module 实例作为 backbone 时，必须显式指定 n_features
    if args.model == "padim" and args.n_features is None:
        args.n_features = cfg.get("n_features")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tubu Anomalib Prediction (Multi-model)")

    # ===== 必要输入 =====
    parser.add_argument("--train-dir","--t", type=str, required=True,
                        help="Training experiment dir (e.g. results/train_20260423_134550), auto-find .ckpt")
    parser.add_argument("--predict-path","--p", type=str, required=True,
                        help="Image or folder to predict")

    # ---- 模型选择（必须与训练时一致）----
    parser.add_argument("--model", type=str, default="patchcore",
                        choices=["patchcore", "simplenet", "rd", "padim"],
                        help="模型类型，必须与训练时使用的 --model 一致！")

    # 模型结构（需与训练一致，一般不用动——会根据 --model 自动适配）
    parser.add_argument("--backbone", type=str, default="resnet18",
                        help="骨干网络（默认根据 --model 自动适配）")
    parser.add_argument("--layers", type=str, nargs="+", default=["layer2", "layer3"],
                        help="特征提取层（默认根据 --model 自动适配）")
    parser.add_argument("--pre-trained",
                        action=argparse.BooleanOptionalAction, default=True)

    # ===== 可调参数 =====
    parser.add_argument("--pixel-sensitivity", type=float, default=0.5,
                        help="Pixel sensitivity (0~1). Higher = larger pred_mask area. Default 0.5")
    parser.add_argument("--image-sensitivity", type=float, default=0.5,
                        help="Image sensitivity (0~1). Default 0.5")

    # 模型特有参数
    parser.add_argument("--coreset-sampling-ratio", type=float, default=0.1,
                        help="[PatchCore] 核心集采样率")
    parser.add_argument("--perlin-threshold", type=float, default=0.2,
                        help="[SimpleNet] Perlin 噪声阈值")
    parser.add_argument("--n-features", type=int, default=None,
                        help="[PaDiM] 降维后保留的特征数 (resnet18=100)")
    parser.add_argument("--input-size", type=int, nargs=2, default=[256, 256],
                        metavar=("H", "W"), help="[RD] 输入图像尺寸")
    parser.add_argument("--anomaly-map-mode", type=str, default="add",
                        choices=["add", "multiply"], help="[RD] 异常图模式")

    # 支持 MMDetection 自定义权重推理
    parser.add_argument("--mmdet-weights", action="store_true",
                        help="若训练时使用了 mmdet 权重，请带上此 Flag")

    # 预训练权重路径（与 train.py 保持一致，默认自动检测 weights/resnet18.pth）
    parser.add_argument("--weights-path", type=str, default=r"weights/resnet18.pth",
                        help="[Timm本地] 本地预训练权重路径。若文件存在则加载（与训练一致），不存在则跳过")

    # 硬件
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--devices", type=int, default=1)

    args = parser.parse_args()
    _apply_model_defaults(args)
    return args


if __name__ == "__main__":
    main()
