from __future__ import annotations

import os

# 禁止 timm/huggingface 在线下载（推理阶段完全不需要，避免网络超时）
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import logging
import re
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

    # ---- AnomalyDINO 模式 ----
    if args.model == "anomaly_dino":
        return _load_dino_weights(args, log)

    # ---- RD 模型不支持自定义权重注入（decoder 需要字符串查表）----
    if args.model == "rd":
        if args.mmdet_weights or (args.weights_path and Path(args.weights_path).is_file()):
            log.warning(f"  ⚠ RD 不支持自定义权重注入（decoder 限制），已忽略")
            log.warning(f"     将使用 timm 在线预训练的 {args.backbone}")
        log.info(f"  预训练权重: RD 使用 timm 在线预训练 ({args.backbone})")
        return backbone_arg

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


def _load_dino_weights(args: argparse.Namespace, log: logging.Logger):
    """
    加载 DINOv2/DINOv3 权重，返回 (feature_encoder, dino_config)。
    逻辑与 train.py 完全一致，确保 checkpoint 的 state_dict 键名对齐。
    """
    dino_path = args.dino_weights

    # ---- 无自定义权重 ----
    if not dino_path:
        log.info(f"  DINO 预训练权重: 使用官方在线/缓存 ({args.encoder_name})")
        return None  # 让 AnomalyDINO 内部通过 DinoV2Loader 处理

    dino_path = Path(dino_path)
    if not dino_path.is_file():
        log.error(f"❌ 找不到 DINO 权重文件: {dino_path.resolve()}")
        sys.exit(1)

    fname = dino_path.name.lower()
    if "dinov3" in fname or fname.startswith("dinov3"):
        dino_ver = "dinov3"
    else:
        dino_ver = "dinov2"

    log.info(f"  DINO 权重           : {dino_path.name}")
    log.info(f"  DINO 版本           : {dino_ver}")
    log.info(f"  Encoder 名称        : {args.encoder_name}")

    encoder_name = args.encoder_name
    feature_encoder = _load_dino_encoder(encoder_name, dino_path, dino_ver, log)
    return {"type": "dino", "path": str(dino_path), "version": dino_ver,
            "encoder_name": encoder_name, "feature_encoder": feature_encoder}


def _load_dino_encoder(encoder_name: str, weight_path: Path, dino_ver: str,
                       log: logging.Logger):
    """加载 DINOv2/DINOv3 编码器并灌入本地权重。与 train.py 完全一致。"""
    log.info(f"  加载 DINO{'' if dino_ver == 'dinov2' else '3'} 权重: {weight_path.name}")

    if dino_ver == "dinov3":
        return _load_dinov3_encoder(encoder_name, weight_path, log)

    # ---- DINOv2 路径 ----
    from anomalib.models.components.dinov2 import DinoV2Loader

    loader = DinoV2Loader()
    model_type, architecture, patch_size = loader._parse_name(encoder_name)
    try:
        encoder = loader.create_model(model_type, architecture, patch_size)
        log.info(f"  DINOv2 空架构创建成功: {model_type} / {architecture} / p{patch_size}")
    except (ValueError, KeyError) as e:
        log.warning(f"  DinoV2Loader 不支持 '{encoder_name}' ({e})，尝试手动构建...")
        encoder = _build_fallback_dino_encoder(encoder_name, log)

    state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)

    missing, unexpected = [], []
    try:
        result = encoder.load_state_dict(state_dict, strict=True)
        missing, unexpected = result.missing_keys, result.unexpected_keys
    except RuntimeError:
        log.warning("  ⚠ strict 匹配失败，使用 strict=False")
        result = encoder.load_state_dict(state_dict, strict=False)
        missing, unexpected = result.missing_keys, result.unexpected_keys

    matched = len(state_dict) - len(missing)
    log.info(f"  ✅ 匹配 Tensor: {matched}/{len(state_dict)}")
    if missing:
        log.info(f"     缺失键: {len(missing)} 个 (首次出现: {missing[:3]})")
    if unexpected:
        log.info(f"     多余键: {len(unexpected)} 个")
    return encoder


def _load_dinov3_encoder(encoder_name: str, weight_path: Path, log: logging.Logger):
    """加载 DINOv3 编码器（架构与 v2 完全不同）。"""
    import sys as _sys

    dinov3_dir = Path(r"Z:\14-调试数据\lxm\Projects\DINOv3")
    if not (dinov3_dir / "hub").is_dir():
        log.error(f"❌ DINOv3 项目目录不存在: {dinov3_dir.resolve()}")
        _sys.exit(1)

    if str(dinov3_dir) not in _sys.path:
        _sys.path.insert(0, str(dinov3_dir))

    try:
        from hub.backbones import _DINOV2_BASE_WITH_REGISTRY
    except ImportError:
        from hub.backbones import _DINOV2_BASE_WITH_REGISTRY  # noqa: F811

    model_fn = getattr(_DINOV2_BASE_WITH_REGISTRY, encoder_name, None)
    if model_fn is None:
        available = [k for k in dir(_DINOV2_BASE_WITH_REGISTRY) if k.startswith("dinov3")]
        log.error(f"❌ DINOv3 hub 中没有 '{encoder_name}'。可用: {available}")
        _sys.exit(1)

    model = model_fn()
    sd = torch.load(weight_path, map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=False)
    log.info(f"  ✅ DINOv3 权重加载成功: {encoder_name} ({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M)")
    return model


def _build_fallback_dino_encoder(encoder_name: str, log: logging.Logger):
    """当 DinoV2Loader 不支持某 encoder_name 时，回退到手动构建 ViT。"""
    from anomalib.models.components.dinov2 import vision_transformer as dinov2_models

    arch_map = {
        "s": "small",   "small": "small",
        "b": "base",    "base": "base",
        "l": "large",   "large": "large",
    }
    m = re.search(r'vit([sbl]|small|base|large)(\d+)(plus)?', encoder_name, re.IGNORECASE)
    if not m:
        raise ValueError(f"无法从 '{encoder_name}' 解析出 ViT 架构")

    arch_short = m.group(1).lower()
    patch_size = int(m.group(2))
    architecture = arch_map.get(arch_short)
    if not architecture:
        raise ValueError(f"未知的 ViT 架构标识: '{arch_short}'")

    ctor_name = f"vit_{architecture}"
    ctor = getattr(dinov2_models, ctor_name, None)
    if ctor is None:
        raise ValueError(f"dinov2_models 没有 {ctor_name}")

    log.info(f"  手动构建: {ctor_name}(patch_size={patch_size})")
    return ctor(patch_size=patch_size)


def _auto_load_config(args: argparse.Namespace, log: logging.Logger) -> None:
    """
    从 train_dir/train_config.json 自动加载训练配置，覆盖 predict 默认值。
    CLI 显式传入的参数优先级高于 config（用户手动指定则不覆盖）。
    """
    import json

    train_dir = Path(args.train_dir)
    config_path = train_dir / "train_config.json"

    if not config_path.is_file():
        log.warning(f"  ⚠ 未找到 {config_path.name}，使用 CLI 参数 / 默认值")
        log.info("     提示：重新训练一次即可自动生成配置文件")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # 记录哪些 key 被覆盖了
    _overridden = []
    _skip_keys = {"model", "backbone", "layers", "image_size", "input_size",
                  "mmdet_weights", "weights_path", "pre_trained",
                  "n_features", "coreset_sampling_ratio", "perlin_threshold",
                  "anomaly_map_mode", "dino_weights", "encoder_name",
                  "masking", "coreset_subsampling"}

    for key in _skip_keys:
        if key not in config:
            continue
        config_val = config[key]
        # CLI 默认值 → 用 config 覆盖；CLI 用户显式指定 → 保持不变
        # 判断方式：对比当前 args 值与 parser 默认值是否相同

        # 特殊处理：layers 是 list，默认 ["layer2","layer3"]
        if key == "layers" and args.layers == ["layer2", "layer3"]:
            args.layers = config_val
            _overridden.append(key)
        elif getattr(args, key, None) == _DEFAULTS.get(key):
            setattr(args, key, config_val)
            _overridden.append(key)

    if _overridden:
        log.info(f"  📋 从 train_config.json 自动加载:")
        for k in _overridden:
            v = config[k]
            log.info(f"     --{k.replace('_', '-')} = {v}")
    else:
        log.info(f"  📋 已读取 {config_path.name}，所有参数与 CLI 一致")


# predict.py 的 parser 默认值（用于判断用户是否显式指定）
_DEFAULTS = {
    "model": "patchcore",
    "backbone": "resnet18",
    "layers": ["layer2", "layer3"],
    "image_sensitivity": 0.5,
    "pixel_sensitivity": 0.5,
    "coreset_sampling_ratio": 0.1,
    "perlin_threshold": 0.2,
    "n_features": None,
    "anomaly_map_mode": "add",
    "input_size": [256, 256],
    "pre_trained": True,
    "weights_path": r"weights/timm/resnet18.pth",
    "mmdet_weights": None,
    "dino_weights": None,
    "encoder_name": "dinov2_vit_small_14",
    "masking": False,
    "coreset_subsampling": True,
}


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

    # ---- 从 train_dir 自动加载训练配置（自适应）----
    _auto_load_config(args, log)

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

    use_pretrained = (args.weights_path is None and not args.mmdet_weights and args.dino_weights is None) and args.pre_trained
    common_kwargs = dict(
        visualizer=ImageVisualizer(output_dir=images_dir),
        post_processor=post_processor,
    )

    if args.model == "anomaly_dino":
        model = _create_anomaly_dino(args, backbone_arg, images_dir, post_processor, log)
    elif args.model == "patchcore":
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
        log.error(f"❌ 不支持的模型: {args.model} (可选: patchcore/simplenet/rd/padim/anomaly_dino)")
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


def _create_anomaly_dino(args, dino_config, images_dir: Path,
                          post_processor, log: logging.Logger):
    """创建 AnomalyDINO 模型（推理用），与 train.py 逻辑一致。"""
    from anomalib.models.image.anomaly_dino.lightning_model import AnomalyDINO
    from anomalib.visualization.image.visualizer import ImageVisualizer

    encoder_name = args.encoder_name
    masking = args.masking
    coreset_subsampling = args.coreset_subsampling
    sampling_ratio = args.coreset_sampling_ratio

    vis_kwarg = ImageVisualizer(output_dir=images_dir) if images_dir else False
    if not images_dir:
        log.info("  ImageVisualizer: 禁用")

    log.info(f"  encoder_name         : {encoder_name}")
    log.info(f"  masking              : {masking}")
    log.info(f"  coreset_subsampling  : {coreset_subsampling}")
    log.info(f"  sampling_ratio       : {sampling_ratio}")

    # ---- 自定义 DINO 权重注入 ----
    if isinstance(dino_config, dict) and dino_config.get("type") == "dino":
        feature_encoder = dino_config["feature_encoder"]
        # dummy 名称通过内部校验，实际 feature_encoder 已被替换
        model = AnomalyDINO(
            num_neighbours=1,
            encoder_name="dinov2_vit_small_14",
            masking=masking,
            coreset_subsample=False,
            sampling_ratio=sampling_ratio,
            visualizer=vis_kwarg,
            post_processor=post_processor,
        )
        model.model.feature_encoder = feature_encoder
        log.info(f"  ✅ DINO 自定义权重已注入")
        return model

    # ---- 无自定义权重 ----
    log.info(f"  使用 DINOv2 官方/缓存权重")
    return AnomalyDINO(
        num_neighbours=1,
        encoder_name=encoder_name,
        masking=masking,
        coreset_subsample=coreset_subsampling,
        sampling_ratio=sampling_ratio,
        visualizer=vis_kwarg,
        post_processor=post_processor,
    )


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
                        choices=["patchcore", "simplenet", "rd", "padim", "anomaly_dino"],
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
    parser.add_argument("--mmdet-weights", type=str, default=None,
                        help="[MMDetection] MMDetection 权重路径。若训练时使用了此项，必须一致！")

    # 预训练权重路径 (Timm / MMDet / DINO 三选一)
    weight_group = parser.add_mutually_exclusive_group()
    weight_group.add_argument("--weights-path", type=str, default=r"weights/timm/resnet18.pth",
                        help="[Timm本地] 本地预训练权重路径。设空字符串''跳过")
    weight_group.add_argument("--dino-weights", type=str, default=None,
                        help="[DINOv2/DINOv3] DINO 预训练权重 .pth 路径。配合 --model anomaly_dino 使用。")

    # DINO 特有参数 (anomaly_dino 模型专用)
    parser.add_argument("--encoder-name", type=str, default="dinov2_vit_small_14",
                        help="[AnomalyDINO] DINO 编码器名称 (dinov2_vit_small_14 / dinov2_vit_base_14 / dinov3_vits16 等)")
    parser.add_argument("--masking", action=argparse.BooleanOptionalAction, default=False,
                        help="[AnomalyDINO] 是否启用 PCA 掩码抑制背景特征")
    parser.add_argument("--coreset-subsampling", action=argparse.BooleanOptionalAction, default=True,
                        help="[AnomalyDINO] 是否启用 coreset 降采样")

    # 硬件
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--devices", type=int, default=1)

    args = parser.parse_args()
    _apply_model_defaults(args)
    return args


if __name__ == "__main__":
    main()
