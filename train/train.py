from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import torch


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


def _add_src_to_path() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    src = repo_root / "src"
    sys.path.insert(0, str(src))


class _StageLogger:
    """带阶段的日志器，自动记录每个训练阶段的时间戳和耗时。"""

    def __init__(self, experiment_dir: Path):
        self.logger = self._setup(experiment_dir)
        self._stage_start: float | None = None

    def _setup(self, exp_dir: Path) -> logging.Logger:
        log = logging.getLogger("anomalib_experiment")
        log.setLevel(logging.DEBUG)
        fmt = logging.Formatter("[%(asctime)s] %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

        # 终端
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logging.INFO)
        sh.setFormatter(fmt)
        log.addHandler(sh)

        # 文件（DEBUG 级别）
        fh = logging.FileHandler(exp_dir / "train.log", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        log.addHandler(fh)

        return log

    def header(self, title: str) -> None:
        self.logger.info("=" * 60)
        self.logger.info(f"  {title}")
        self.logger.info("=" * 60)

    def stage(self, name: str) -> None:
        """标记一个新阶段开始。"""
        now = time.time()
        if self._stage_start is not None:
            elapsed = now - self._stage_start
            self.logger.info(f"[DONE] 耗时 {elapsed:.1f}s")
        self._stage_start = now
        self.logger.info(f"\n{'─' * 40}")
        self.logger.info(f">>> 阶段: {name}")
        self.logger.info(f"{'─' * 40}")

    def done(self) -> None:
        """结束当前阶段计时。"""
        if self._stage_start is not None:
            elapsed = time.time() - self._stage_start
            self.logger.info(f"[DONE] 耗时 {elapsed:.1f}s")
            self._stage_start = None

    # 代理常用方法
    def info(self, msg: str) -> None:
        self.logger.info(msg)

    def warning(self, msg: str) -> None:
        self.logger.warning(msg)

    def error(self, msg: str) -> None:
        self.logger.error(msg)


def main(args: argparse.Namespace | None = None) -> None:
    _add_src_to_path()

    from anomalib.callbacks import ModelCheckpoint
    from anomalib.data import Folder
    from anomalib.engine import Engine
    from anomalib.post_processing import PostProcessor
    from anomalib.utils.checkpoint_io import FileCheckpointIO
    from anomalib.visualization import ImageVisualizer

    args = _parse_args() if args is None else args

    # ---- 实验目录 ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = args.exp_name or f"train_{timestamp}"
    results_root = Path(r"results") / exp_name
    weights_dir = results_root / "weights" / "lightning"
    images_dir = results_root / "images"
    results_root.mkdir(parents=True, exist_ok=True)

    log = _StageLogger(results_root)
    log.header(f"Experiment: {exp_name}")
    log.info(f"结果目录 : {results_root.resolve()}")

    t_total = time.time()

    # ================================================================
    # 阶段 1：加载数据集 & 打印摘要
    # ================================================================
    log.stage("1. 加载数据集")

    dataset_root = Path(args.dataset_root)
    log.info(f"数据集根目录: {dataset_root}")

    datamodule = Folder(
        name="tubu_loujinshu",
        root=dataset_root,
        normal_dir=args.normal_dir,
        abnormal_dir=args.abnormal_dir,
        mask_dir=args.mask_dir,
        extensions=tuple(args.extensions),
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        test_split_mode="from_dir",
        test_split_ratio=args.test_split_ratio,
        val_split_mode="same_as_test",
        val_split_ratio=args.val_split_ratio,
        seed=args.seed,
    )

    # 打印数据集统计
    datamodule.setup()
    train_dl = datamodule.train_dataloader()
    val_dl = datamodule.val_dataloader()
    test_dl = datamodule.test_dataloader()

    log.info(f"  正常图 (train): {len(train_dl.dataset)} 张")
    log.info(f"  验证集   (val) : {len(val_dl.dataset)} 张")
    log.info(f"  测试集  (test) : {len(test_dl.dataset)} 张")
    log.info(f"  图像尺寸       : {args.image_size}x{args.image_size}")
    log.info(f"  batch_size      : train={args.train_batch_size}, eval={args.eval_batch_size}")

    log.done()

    # ================================================================
    # 阶段 2：构建模型
    # ================================================================
    log.stage("2. 构建模型")

    post_processor = PostProcessor(
        image_sensitivity=args.image_sensitivity,
        pixel_sensitivity=args.pixel_sensitivity,
    )
    log.info(f"  模型                : {args.model}")
    if args.model == "anomaly_dino":
        log.info(f"  encoder_name         : {args.encoder_name}")
        log.info(f"  masking              : {args.masking}")
        log.info(f"  coreset_subsampling  : {args.coreset_subsampling}")
    else:
        log.info(f"  backbone            : {args.backbone}")
        log.info(f"  layers              : {args.layers}")
    log.info(f"  image_size          : {args.image_size}")
    log.info(f"  max_epochs          : {args.max_epochs} "
             f"({'特征建库' if args.max_epochs == 1 else '梯度训练'})")
    _use_fp16 = args.model in ("simplenet", "rd")
    log.info(f"  precision           : {'FP16 (混合精度)' if _use_fp16 else 'FP32 (Memory Bank 方法无需梯度)'}")
    log.info(f"  vis_samples         : {args.vis_samples} ({'跳过' if args.vis_samples == 0 else '训练后随机采样'})")
    if args.no_vis_test:
        log.info(f"  test_visualize      : OFF (不输出图片，只算指标)")
    else:
        log.info(f"  test_visualize      : ON (输出到 images/ 目录)")

    # ---- 预训练权重加载（返回 backbone_arg 或 None）----
    backbone_arg = _load_backbone_weights(args, log)

    # ---- 根据模型类型实例化 ----
    _test_images_dir = None if args.no_vis_test else images_dir
    model = _create_model(args, backbone_arg, _test_images_dir, post_processor, log)

    checkpoint_callback = ModelCheckpoint(
        dirpath=weights_dir,
        filename="model",
        auto_insert_metric_name=False,
        save_last=args.save_last,
        save_top_k=args.save_top_k,
        save_weights_only=args.save_weights_only,
    )

    # Memory Bank 方法（patchcore/padim）无梯度计算，不能开 FP16
    # 训练式方法（simplenet/rd）需要梯度，开 FP16 节省显存
    use_fp16 = args.model in ("simplenet", "rd")
    engine = Engine(
        callbacks=[checkpoint_callback],
        default_root_dir=results_root,
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.max_epochs,
        precision="16-mixed" if use_fp16 else "32",
        plugins=[FileCheckpointIO()],
    )

    log.done()

    # ================================================================
    # 阶段 3a：训练 (fit) — 提取 embedding + coreset 采样
    # ================================================================
    log.stage("3a. 训练 (特征提取)")

    try:
        engine.fit(model=model, datamodule=datamodule)
    except Exception as e:
        log.error(f"训练失败: {type(e).__name__}: {e}")
        import traceback
        log.error(traceback.format_exc())
        sys.exit(1)

    log.info("  ✅ 训练完成")
    log.done()

    # ================================================================
    # 阶段 3b：测试 (test) — 在验证/测试集上评估
    # ================================================================
    log.stage("3b. 测试 (模型评估)")

    test_results = None
    if args.skip_test:
        log.info("  ⏭️ 跳过测试 (--skip-test)")
    else:
        try:
            test_results = engine.test(model=model, datamodule=datamodule)
        except torch.cuda.OutOfMemoryError:
            log.error("❌ CUDA 显存不足！测试阶段 OOM")
            log.error("   建议：减小 --eval-batch-size (如改为 1 或 2)，或使用 --skip-test 跳过")
        except Exception as e:
            log.error(f"测试失败: {type(e).__name__}: {e}")
            import traceback as tb
            log.error(tb.format_exc())
        else:
            log.done()

    # ---- 打印测试指标 ----
    if test_results:
        log.info("\n--- 测试指标 ---")
        for k, v in test_results[0].items():
            if isinstance(v, float):
                log.info(f"  {k}: {v:.6f}")
            else:
                log.info(f"  {k}: {v}")
    else:
        log.warning("未返回测试结果（可能测试阶段失败）")

    # ---- 检查 checkpoint ----
    ckpt_path_str = engine.checkpoint_callback.last_model_path
    ckpt_path = Path(ckpt_path_str) if ckpt_path_str else None
    log.info(f"\nCheckpoint: {ckpt_path}")
    if ckpt_path and ckpt_path.is_file():
        size_mb = ckpt_path.stat().st_size / (1024 * 1024)
        log.info(f"  文件大小: {size_mb:.1f} MB")
    elif ckpt_path:
        log.error(f"  ❌ Checkpoint 不存在！{ckpt_path}")

    # ================================================================
    # 阶段 4：采样预测可视化
    # ================================================================
    if ckpt_path and ckpt_path.is_file() and args.vis_samples > 0:
        log.stage("4. 采样预测可视化")
        _sample_and_predict(model, engine, ckpt_path, dataset_root, args.abnormal_dir,
                            args.extensions, images_dir, log, seed=args.seed,
                            max_samples=args.vis_samples)
        log.done()
    elif args.vis_samples == 0:
        log.info("\n⏭️ 跳过采样预测 (--vis-samples=0，需要时跑 predict.py)")
    else:
        log.warning("\n跳过采样预测（无有效 checkpoint）")

    # ================================================================
    # 保存训练配置（供 predict.py 自适应读取）
    # ================================================================
    import json
    _config_fields = [
        "model", "backbone", "layers", "image_size", "input_size",
        "mmdet_weights", "weights_path", "dino_weights", "pre_trained",
        "n_features", "coreset_sampling_ratio", "perlin_threshold",
        "anomaly_map_mode", "image_sensitivity", "pixel_sensitivity",
        "no_vis_test", "vis_samples",
        # AnomalyDINO 特有
        "encoder_name", "masking", "coreset_subsampling",
    ]
    train_config = {f: getattr(args, f) for f in _config_fields}
    config_path = results_root / "train_config.json"
    with open(config_path, "w", encoding="utf-8") as _f:
        json.dump(train_config, _f, indent=2, ensure_ascii=False)
    log.info(f"  配置文件: {config_path}")

    # ================================================================
    # 总结
    # ================================================================
    total_elapsed = time.time() - t_total
    log.header(f"全部完成！总耗时 {total_elapsed:.1f}s ({total_elapsed / 60:.1f}min)")
    log.info(f"  结果目录: {results_root.resolve()}")
    log.info(f"  权重文件: {ckpt_path}")
    log.info(f"  可视化图: {'禁用 (--no-vis-test)' if args.no_vis_test else images_dir}")
    log.info(f"  日志文件: {results_root / 'train.log'}")


def _sample_and_predict(model, engine, ckpt_path: Path, dataset_root: Path,
                        abnormal_dir: str, extensions: list[str],
                        images_dir: Path, log: _StageLogger, seed: int = 42,
                        sample_ratio: float = 0.1, max_samples: int = 100) -> None:
    """训练结束后，从异常图中随机抽样进行预测可视化。"""
    rng = random.Random(seed)

    abnormal_dir_path = dataset_root / abnormal_dir
    if not abnormal_dir_path.is_dir():
        log.warning(f"异常图目录不存在: {abnormal_dir_path}，跳过")
        return

    ext_set = tuple(set(extensions))
    all_images = sorted([
        p for p in abnormal_dir_path.rglob("*")
        if p.is_file() and p.suffix.lower() in ext_set
    ])
    total = len(all_images)
    if total == 0:
        log.warning(f"异常图目录下无图片: {abnormal_dir_path}")
        return

    # 过滤含非打印字符的文件名（Windows 路径不支持零宽空格等 Unicode 字符）
    import string
    _printable = set(string.printable)
    clean_images = []
    skipped_chars = 0
    for p in all_images:
        name = p.name
        non_print = [c for c in name if c not in _printable]
        if non_print:
            skipped_chars += 1
            continue
        clean_images.append(p)

    if skipped_chars > 0:
        log.warning(f"  跳过 {skipped_chars} 个含非打印字符的文件名 "
                     f"(Windows 路径不支持零宽空格等字符)")

    all_images = clean_images
    total = len(all_images)
    if total == 0:
        log.warning(f"过滤后无可用图片")
        return

    n_samples = min(max(int(total * sample_ratio), 1), max_samples)
    sampled = rng.sample(all_images, n_samples)

    log.info(f"  抽样: {n_samples}/{total} 张 "
             f"(ratio={sample_ratio}, max={max_samples})")

    # 阶段 4 始终需要可视化输出：即使 --no-vis_test 禁用了测试时的 visualizer，
    # 这里也要临时注入一个，否则 predict 不会生成任何图片
    from anomalib.visualization.image.visualizer import ImageVisualizer
    _orig_visualizer = getattr(model, "visualizer", None)
    _vis_dir = images_dir / "samples"
    model.visualizer = ImageVisualizer(output_dir=_vis_dir)
    log.info(f"  可视化目录: {_vis_dir}")

    success, fail = 0, 0
    t0 = time.time()
    for i, img_path in enumerate(sampled, 1):
        try:
            engine.predict(
                model=model,
                data_path=str(img_path),
                return_predictions=False,
            )
            success += 1
            if i % 20 == 0 or i == n_samples:
                log.info(f"  进度: {i}/{n_samples} ({success} 成功, {fail} 失败)")
        except Exception as e:
            fail += 1
            log.warning(f"  [{i}/{n_samples}] 失败 {img_path.name}: {e}")

    # 恢复原始 visualizer 状态
    model.visualizer = _orig_visualizer

    elapsed = time.time() - t0
    log.info(f"  完成: {success} 成功, {fail} 失败, 耗时 {elapsed:.1f}s")


# ================================================================
# 模型默认值适配（根据 --model 自动调整 backbone/layers）
# ================================================================
_MODEL_DEFAULTS = {
    "patchcore": {
        "backbone":   "resnet18",
        "layers":     ["layer2", "layer3"],
        "max_epochs": 1,       # Memory Bank 方法，不需要真正训练
    },
    "simplenet": {
        "backbone":   "resnet18.tv_in1k",  # SimpleNet 要求 .tv 权重格式
        "layers":     ["layer2", "layer3"],
        "max_epochs": 100,    # 需要训练判别头
    },
    "rd": {
        "backbone":   "resnet18",
        "layers":     ["layer1", "layer2", "layer3"],
        "max_epochs": 100,    # 需要训练解码器
    },
    "padim": {
        "backbone":   "resnet18",
        "layers":     ["layer2", "layer3"],   # 15GB 显存安全配置
        "max_epochs": 1,                       # Memory Bank 方法，不需要真正训练
        "n_features": 100,                     # resnet18 推荐 100 (wrn50=550)
    },
    "anomaly_dino": {
        "encoder_name":       "dinov2_vit_small_14",
        "coreset_subsampling": True,
        "masking":             False,
        "image_size":          252,
        "max_epochs":          1,              # Memory Bank 方法，不需要真正训练
    },
}


def _apply_model_defaults(args: argparse.Namespace) -> None:
    """根据 --model 覆盖 backbone/layers/max_epochs/input-size 的默认值（仅当用户未手动指定时生效）。"""
    cfg = _MODEL_DEFAULTS[args.model]
    if args.model == "anomaly_dino":
        # AnomalyDINO 使用 DINO ViT encoder，不使用 CNN backbone/layers
        if args.encoder_name == "dinov2_vit_small_14" and "encoder_name" in cfg:
            args.encoder_name = cfg["encoder_name"]
        if args.image_size == 256 and "image_size" in cfg:
            args.image_size = cfg["image_size"]
            args.input_size = [cfg["image_size"], cfg["image_size"]]
        if not args.coreset_subsampling and "coreset_subsampling" in cfg:
            args.coreset_subsampling = cfg["coreset_subsampling"]
        return

    if args.backbone == "wide_resnet50_2" and args.model != "patchcore":
        args.backbone = cfg["backbone"]
    if args.layers == ["layer2", "layer3"] and args.model in ("rd", "padim"):
        args.layers = cfg["layers"]
    if args.max_epochs == 1 and args.model in ("simplenet", "rd"):
        args.max_epochs = cfg["max_epochs"]
    # n_features 自动适配 (PaDiM: resnet18=100, wrn50=550)
    if args.model == "padim" and args.n_features is None:
        args.n_features = cfg.get("n_features")
    # input-size 跟随 image-size（RD 模型需要）
    if args.input_size == [256, 256] and args.image_size != 256:
        args.input_size = [args.image_size, args.image_size]


def _load_backbone_weights(args: argparse.Namespace, log: _StageLogger):
    """加载预训练权重，返回 backbone_arg (模型实例 / 字符串 / None)。
    
    支持三种模式:
      A) --mmdet-weights  : MMDetection 检测模型权重 (CNN)
      B) --weights-path   : Timm 本地预训练权重 (CNN)
      C) --dino-weights   : DINOv2/DINOv3 自监督权重 (ViT)
    """
    backbone_arg = args.backbone

    # ---- 模式 C: DINOv2/DINOv3 权重 ----
    if args.dino_weights:
        dino_path = Path(args.dino_weights)
        if not dino_path.is_file():
            log.error(f"❌ 找不到 DINO 权重文件: {dino_path.resolve()}")
            sys.exit(1)

        # 从文件名推断 DINO 版本和架构
        fname = dino_path.name.lower()
        if "dinov2" in fname or fname.startswith("dinov2"):
            dino_ver = "dinov2"
        elif "dinov3" in fname or fname.startswith("dinov3"):
            dino_ver = "dinov3"
        else:
            log.error(f"❌ 无法从文件名识别 DINO 版本: {fname} (需包含 dinov2 或 dinov3)")
            sys.exit(1)

        log.info(f"  DINO 权重           : {dino_path.resolve()}")
        log.info(f"  DINO 版本           : {dino_ver}")
        log.info(f"  Encoder 名称        : {args.encoder_name}")

        # 返回 DINO 配置信息供 _create_model 使用
        return {"type": "dino", "path": str(dino_path), "version": dino_ver,
                "encoder_name": args.encoder_name}

    # ---- AnomalyDINO 无自定义权重：使用官方在线下载 ----
    if args.model == "anomaly_dino":
        log.info(f"  预训练权重: DINOv2 官方在线下载 ({args.encoder_name})")
        log.info(f"     (首次运行自动下载到 ~/.cache/anomalib/dinov2/)")
        log.info(f"     如需使用本地权重，请加 --dino-weights <路径>")
        return None  # 让 AnomalyDINO 自己通过 DinoV2Loader 下载

    # ---- RD 模型不支持自定义权重注入（decoder 需要字符串查表）----
    if args.model == "rd":
        if args.mmdet_weights or (args.weights_path and Path(args.weights_path).is_file()):
            log.warning(f"  ⚠ RD 不支持自定义权重注入（decoder 限制），已忽略")
            log.warning(f"     将使用 timm 在线预训练的 {args.backbone}")
        log.info(f"  预训练权重: RD 使用 timm 在线预训练 ({args.backbone})")
        return backbone_arg

    if args.model == "simplenet":
        # SimpleNet 通过 timm 自动加载 .tv 权重，不支持自定义权重注入
        log.info(f"  预训练权重: SimpleNet 使用 timm .tv 格式 ({args.backbone})")
        return backbone_arg

    if args.mmdet_weights:
        # 模式 A: MMDetection 预训练权重
        weights_path = Path(args.mmdet_weights)
        if not weights_path.is_file():
            log.error(f"❌ 找不到 MMDetection 权重文件: {weights_path.resolve()}")
            sys.exit(1)
        log.info(f"  正在加载 MMDet 权重 : {weights_path.resolve()}")
        import torchvision.models as tv_models

        if hasattr(tv_models, args.backbone):
            custom_backbone = getattr(tv_models, args.backbone)()
        else:
            log.error(f"❌ torchvision 中不支持该模型: {args.backbone}")
            sys.exit(1)

        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        backbone_state_dict = {k.replace("backbone.", ""): v
                               for k, v in state_dict.items()
                               if k.startswith("backbone.")}
        if len(backbone_state_dict) == 0:
            log.warning("⚠ 未找到 'backbone.' 前缀，尝试直接使用全部权重匹配...")
            backbone_state_dict = state_dict

        custom_backbone.load_state_dict(backbone_state_dict, strict=False)
        log.info(f"  ✅ MMDet权重灌入成功！匹配 {len(backbone_state_dict)} 个 Tensor。")
        return custom_backbone

    elif args.weights_path:
        # 模式 B: Timm 本地权重
        weights_path = Path(args.weights_path)
        if weights_path.is_file():
            log.info(f"  加载本地预训练权重: {weights_path.resolve()}")
            import timm
            bb_model = timm.create_model(
                args.backbone, pretrained=False,
                features_only=True, exportable=True,
                out_indices=_timm_layer_to_idx(args.backbone, args.layers),
            )
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
            try:
                bb_model.load_state_dict(state_dict, strict=False)
            except RuntimeError as e:
                if "size mismatch" in str(e):
                    log.error(f"  ❌ 权重文件与 backbone 架构不匹配！")
                    log.error(f"     当前 backbone : {args.backbone}")
                    log.error(f"     权重文件      : {weights_path.name}")
                    log.error(f"     尝试用 --backbone 指定与权重文件匹配的架构")
                    log.error(f"     例: --model padim --backbone resnet18 --weights-path {weights_path}")
                    sys.exit(1)
                raise
            log.info(f"  ✅ 本地权重加载成功 (strict=False，允许部分键不匹配)")
            return bb_model
        else:
            log.warning(f"  ⚠ 权重文件不存在: {weights_path}，将使用 timm 在线下载")

    # 模式 D: 在线下载 或 随机初始化
    if args.pre_trained and not args.mmdet_weights:
        log.info("  预训练权重: 从 timm 在线下载 (ImageNet)")
    else:
        log.info("  预训练权重: 不使用 (随机初始化)")

    return backbone_arg


def _create_model(args, backbone_arg, images_dir: Path,
                  post_processor, log: _StageLogger):
    """根据 --model 参数创建对应的异常检测模型实例。"""
    from anomalib.models.image.patchcore.lightning_model import Patchcore
    from anomalib.models.image.supersimplenet.lightning_model import Supersimplenet
    from anomalib.models.image.reverse_distillation.lightning_model import ReverseDistillation
    from anomalib.models.image.padim.lightning_model import Padim
    from anomalib.visualization.image.visualizer import ImageVisualizer

    # ---- AnomalyDINO (DINOv2/DINOv3 ViT backbone) ----
    if args.model == "anomaly_dino":
        return _create_anomaly_dino(args, backbone_arg, images_dir, post_processor, log)

    use_pretrained = (args.weights_path is None and not args.mmdet_weights) and args.pre_trained
    # --no-vis-test: 显式传 visualizer=False → 模型内部不会创建 ImageVisualizer callback
    if images_dir is None:
        common_kwargs = dict(visualizer=False, post_processor=post_processor)
        log.info("  ImageVisualizer: 禁用（--no-vis-test，visualizer=False）")
    else:
        common_kwargs = dict(
            visualizer=ImageVisualizer(output_dir=images_dir),
            post_processor=post_processor,
        )

    if args.model == "patchcore":
        log.info(f"  coreset_sampling_ratio: {args.coreset_sampling_ratio}")
        log.info(f"  pixel_threshold      : {1.0 - args.pixel_sensitivity:.2f} "
                 f"(sensitivity={args.pixel_sensitivity})")
        return Patchcore(
            backbone=backbone_arg,
            layers=tuple(args.layers),
            pre_trained=use_pretrained,
            coreset_sampling_ratio=args.coreset_sampling_ratio,
            **common_kwargs,
        )

    elif args.model == "simplenet":
        log.info(f"  perlin_threshold     : {args.perlin_threshold}")
        return Supersimplenet(
            backbone=backbone_arg,       # 必须是 .tv_in1k 结尾
            layers=args.layers,
            perlin_threshold=args.perlin_threshold,
            **common_kwargs,
        )

    elif args.model == "rd":
        log.info(f"  anomaly_map_mode     : {args.anomaly_map_mode}")
        return ReverseDistillation(
            backbone=backbone_arg,
            layers=tuple(args.layers),
            pre_trained=use_pretrained,
            anomaly_map_mode=args.anomaly_map_mode,
            **common_kwargs,
        )

    elif args.model == "padim":
        n_feat = args.n_features
        log.info(f"  n_features           : {n_feat or 'auto'}")
        return Padim(
            backbone=backbone_arg,
            layers=args.layers,
            pre_trained=use_pretrained,
            n_features=n_feat,
            **common_kwargs,
        )

    else:
        log.error(f"❌ 不支持的模型: {args.model} (可选: patchcore/simplenet/rd/padim/anomaly_dino)")
        sys.exit(1)


def _create_anomaly_dino(args, dino_config, images_dir: Path,
                          post_processor, log: _StageLogger):
    """创建 AnomalyDINO 模型，支持自定义 DINOv2/DINOv3 权重注入。"""
    from anomalib.models.image.anomaly_dino.lightning_model import AnomalyDINO
    from anomalib.models.components.dinov2 import DinoV2Loader
    from anomalib.visualization.image.visualizer import ImageVisualizer

    encoder_name = args.encoder_name
    masking = args.masking
    coreset_subsampling = args.coreset_subsampling
    sampling_ratio = args.coreset_sampling_ratio

    log.info(f"  encoder_name         : {encoder_name}")
    log.info(f"  masking              : {masking}")
    log.info(f"  coreset_subsampling  : {coreset_subsampling}")
    log.info(f"  sampling_ratio       : {sampling_ratio}")

    vis_kwarg = False if images_dir is None else ImageVisualizer(output_dir=images_dir)
    if images_dir is None:
        log.info("  ImageVisualizer: 禁用（--no-vis-test）")

    # ---- 自定义 DINO 权重注入 ----
    if isinstance(dino_config, dict) and dino_config.get("type") == "dino":
        dino_path = Path(dino_config["path"])
        dino_ver = dino_config["version"]

        try:
            feature_encoder = _load_dino_encoder(encoder_name, dino_path, dino_ver, log)
        except Exception as e:
            log.error(f"❌ DINO 权重加载失败: {e}")
            import traceback as tb
            log.error(tb.format_exc())
            sys.exit(1)

        log.info(f"  ✅ 自定义 DINO 权重加载成功！")
        # 临时替换 DinoV2Loader.load，返回已加载好的 encoder，阻止 AnomalyDINO 内部再次下载
        _original_load = DinoV2Loader.load
        def _fake_load(self, model_name):
            return feature_encoder
        DinoV2Loader.load = _fake_load
        try:
            model = AnomalyDINO(
                num_neighbours=1,
                encoder_name=encoder_name,
                masking=masking,
                coreset_subsampling=coreset_subsampling,
                sampling_ratio=sampling_ratio,
                visualizer=vis_kwarg,
                post_processor=post_processor,
            )
        finally:
            DinoV2Loader.load = _original_load
    else:
        # 无自定义权重，让 AnomalyDINO 自己通过 DinoV2Loader 在线下载
        log.info(f"  使用 DINOv2 官方权重（在线下载或缓存）")
        model = AnomalyDINO(
            num_neighbours=1,
            encoder_name=encoder_name,
            masking=masking,
            coreset_subsampling=coreset_subsampling,
            sampling_ratio=sampling_ratio,
            visualizer=vis_kwarg,
            post_processor=post_processor,
        )

    # ===================================================================
    #  AnomalyDINO Memory Bank 建库安全网
    #
    #  根因: feature_encoder.eval() -> 整体被 Lightning 设为 eval (199 modules)
    #        -> forward() self.training=False -> 特征不进 embedding_store
    #  方案: types.MethodType 替换三个方法，绕过 self.training
    # ===================================================================
    import types as _types
    import torch.nn.functional as _F
    import torch as _T

    _tm = model.model  # 闭包引用 AnomalyDINOModel 实例

    def _new_training_step(self, batch, *args, **kwargs):
        """直接提取特征到 embedding_store，不依赖 self.training。"""
        del args, kwargs
        if not _tm.training:
            _tm.train()
            print("[SAFETY-NET] forced tm.train()", flush=True)

        img = batch["image"] if isinstance(batch, dict) else batch.image

        input_tensor = img.type(_tm.memory_bank.dtype)
        _, _, h, w = input_tensor.shape
        ps = _tm.feature_encoder.patch_size
        ch, cw = h % ps, w % ps
        if ch > 0 or cw > 0:
            input_tensor = input_tensor[:, :, ch//2:h-ch+ch//2, cw//2:w-cw+cw//2]
        grid = ((h-ch)//ps, (w-cw)//ps)
        dev = input_tensor.device

        with _T.inference_mode():
            feats = _tm.feature_encoder.get_intermediate_layers(input_tensor, n=1)[0]

        # 保存全量特征作为 masking 回退
        feats_full = feats.clone() if _tm.masking else None

        if _tm.masking:
            import numpy as _np
            masks_np = type(_tm).compute_background_masks(feats.detach().cpu().numpy(), grid)
            masks = _T.from_numpy(masks_np).to(dev)
        else:
            masks = _T.ones(feats.shape[:2], dtype=_T.bool, device=dev)

        feats = feats[masks]

        # 回退: 如果 masking 把所有 patch 都滤掉了，降级为使用全量特征
        if feats.size(0) == 0 and feats_full is not None:
            print(f"[SAFETY-NET] MASKING filtered ALL patches! Falling back to full features.", flush=True)
            feats = feats_full.reshape(-1, feats_full.shape[-1])  # [B, N, D] → [B*N, D]

        feats = _F.normalize(feats, p=2, dim=1)
        _tm.embedding_store.append(feats)

        print(f"[SAFETY-NET] store={len(_tm.embedding_store)} last={feats.shape}", flush=True)
        return _T.tensor(0.0, requires_grad=True, device=input_tensor.device)

    model.training_step = _types.MethodType(_new_training_step, model)

    def _new_epoch_end(self):
        ne = len(_tm.embedding_store); mbn = _tm.memory_bank.numel()
        print(f"[SAFETY-NET] epoch_end: store={ne} bank_n={mbn}", flush=True)
        if ne > 0 and mbn == 0:
            print(f"[SAFETY-NET] fit() with {ne} chunks...", flush=True)
            _tm.fit()
            print(f"[SAFETY-NET] fit() OK: {_tm.memory_bank.shape}", flush=True)
        elif ne == 0:
            print("[SAFETY-NET] ERROR empty store!", flush=True)

    model.on_train_epoch_end = _types.MethodType(_new_epoch_end, model)

    def _new_val_start(self):
        if _tm.memory_bank.numel() == 0:
            ne = len(_tm.embedding_store)
            if ne > 0:
                print(f"[SAFETY-NET] EMERGENCY fit: {ne} chunks", flush=True); _tm.fit()
            else:
                print("[SAFETY-NET] FATAL no features!", flush=True)

    model.on_validation_start = _types.MethodType(_new_val_start, model)
    print("[SAFETY-NET] hooks OK", flush=True)
    return model
    import logging as _logging
    _mb_log = _logging.getLogger(__name__)

    # ---- 1. 包装 training_step：强制 train 模式 ----
    _orig_training_step = model.training_step

    def _safe_training_step(self, batch, *args, **kwargs):
        """确保 AnomalyDINOModel 在训练步骤中处于 train 模式。"""
        torch_model = self.model
        # 显式设为 train 模式（解决 self.training=False 的问题）
        was_training = torch_model.training
        if not was_training:
            torch_model.train()
            _mb_log.warning(
                "[安全网] torch_model.training=%s, 已强制设为 train()",
                was_training,
            )
        try:
            result = _orig_training_step(self, batch, *args, **kwargs)
            # 诊断: 打印收集状态
            n_stored = len(torch_model.embedding_store)
            if n_stored > 0:
                last_shape = torch_model.embedding_store[-1].shape
                _mb_log.debug("[安全网] training_step 完成, 累计 %d 个特征块 (最新 %s)", n_stored, last_shape)
            else:
                _mb_log.warning(
                    "[安全网] ⚠️ training_step 执行后 embedding_store 仍为空! "
                    "self.training=%s, feature_encoder=%s",
                    torch_model.training,
                    type(torch_model.feature_encoder).__name__,
                )
            return result
        finally:
            pass  # 不恢复原模式——在整个训练期间保持 train

    model.training_step = _safe_training_step.__get__(model, type(model))

    # ---- 2. 包装 on_train_epoch_end：训练结束后立即建库 ----
    __orig_train_epoch_end = getattr(model, "on_train_epoch_end", None)

    def _safe_on_train_epoch_end(self):
        """训练 epoch 结束后立即构建 Memory Bank。"""
        torch_model = self.model
        n_embed = len(torch_model.embedding_store)
        mb_size = torch_model.memory_bank.numel()

        _mb_log.info(
            "[安全网] on_train_epoch_end: embedding_store=%d, memory_bank=%d",
            n_embed, mb_size,
        )

        if n_embed > 0 and mb_size == 0:
            # 有待建库的特征，立即 fit
            _mb_log.info("[安全网] 发现 %d 个未建库特征块，立即 fit()", n_embed)
            self.fit()
            _mb_log.info(
                "[安全网] fit() 完成, memory_bank size=%s",
                torch_model.memory_bank.shape,
            )
        elif mb_size > 0:
            _mb_log.info("[安全网] Memory Bank 已就绪 (size=%s), 跳过", torch_model.memory_bank.shape)
        elif n_embed == 0:
            _mb_log.error(
                "[安全网] ❌ 训练结束但 embedding_store 为空! "
                "特征可能未在 training_step 中正确收集。",
            )

        if _orig_train_epoch_end is not None:
            _orig_train_epoch_end()

    model.on_train_epoch_end = _safe_on_train_epoch_end.__get__(model, type(model))

    # ---- 3. 包装 on_validation_start：验证前最后防线 ----
    _orig_val_start = getattr(model, "on_validation_start", None)

    def _safe_on_validation_start(self):
        """验证开始前确保 Memory Bank 可用。"""
        torch_model = self.model
        if torch_model.memory_bank.numel() == 0:
            n_embed = len(torch_model.embedding_store)
            if n_embed > 0:
                _mb_log.info(
                    "[安全网] 验证前紧急建库: %d 个特征块", n_embed,
                )
                self.fit()
            else:
                _mb_log.error(
                    "[安全网] ❌ 验证前 Memory Bank 为空且无待建库特征! "
                    "推理将失败。",
                )

        if _orig_val_start is not None:
            _orig_val_start()

    model.on_validation_start = _safe_on_validation_start.__get__(model, type(model))

    _mb_log.info("✅ AnomalyDINO Memory Bank 安全网已安装 (3 hooks)")
    return model


def _load_dino_encoder(encoder_name: str, weight_path: Path, dino_ver: str,
                       log: _StageLogger):
    """加载 DINOv2/DINOv3 编码器模型并灌入本地权重。

    策略：
      - DINOv2: 通过 DinoV2Loader 创建标准 ViT，然后覆盖加载本地权重
      - DINOv3: 使用 DINOv3 项目自带的模型定义 (DinoVisionTransformer)，
        架构与 DINOv2 完全不同 (RoPE / storage_tokens / bias_mask)，不能混用
    """
    log.info(f"  加载 DINO{'' if dino_ver == 'dinov2' else '3'} 权重: {weight_path.name}")

    if dino_ver == "dinov3":
        return _load_dinov3_encoder(encoder_name, weight_path, log)

    # ---- DINOv2 路径 ----
    from anomalib.models.components.dinov2 import DinoV2Loader

    # 只用 create_model() 创建空架构，避免 from_name() 自动下载官方权重
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
        log.warning(f"  ⚠ strict 匹配失败，使用 strict=False")
        result = encoder.load_state_dict(state_dict, strict=False)
        missing, unexpected = result.missing_keys, result.unexpected_keys

    if missing:
        log.info(f"     缺失键: {len(missing)} 个 (首次出现: {missing[:3]})")
    if unexpected:
        log.info(f"     多余键: {len(unexpected)} 个")

    matched = len(state_dict) - len(missing)
    log.info(f"     匹配 Tensor: {matched}/{len(state_dict)}")
    encoder.eval()
    return encoder


def _load_dinov3_encoder(encoder_name: str, weight_path: Path, log: _StageLogger):
    """使用 DINOv3 项目自身的模型定义加载 DINOv3 编码器。

    DINOv3 的架构与 DINOv2 完全不同:
      - RoPE (旋转位置编码) 替代 learnable pos_embed
      - storage_tokens / mask_token / bias_mask 等新组件
      - LayerNormBF16 等特殊层
    必须使用 DinoVisionTransformer 类，不能灌入 DINOv2 模型。
    """
    dinov3_root = Path(r"Z:\14-调试数据\lxm\Projects\DINOv3")

    # encoder_name 即为 DINOv3 hub 函数名: dinov3_vitb16, dinov3_vits16plus 等
    hub_func_name = encoder_name

    import sys
    dinov3_src = str(dinov3_root)
    if dinov3_src not in sys.path:
        sys.path.insert(0, dinov3_src)

    log.info(f"  DINOv3 hub 函数: {hub_func_name}")
    log.info(f"  DINOv3 项目路径: {dinov3_root}")

    # 动态导入对应的 hub 工厂函数 (如 dinov3_vitb16 / dinov3_vits16)
    from dinov3.hub import backbones as hub_module
    hub_fn = getattr(hub_module, hub_func_name, None)
    if hub_fn is None:
        available = [k for k in dir(hub_module) if k.startswith("dinov3_")]
        raise ValueError(
            f"DINOv3 hub 中没有 '{hub_func_name}'。可用: {available}"
        )

    # 调用工厂函数: pretrained=False 先建空壳, weights=本地路径加载权重
    encoder = hub_fn(pretrained=False, weights=str(weight_path))
    log.info(f"  ✅ DINOv3 编码器加载成功！")
    encoder.eval()
    return encoder


def _build_fallback_dino_encoder(encoder_name: str, log: _StageLogger):
    """当 DinoV2Loader 不支持某 encoder_name 时，回退到手动构建 ViT。

    支持的命名格式:
      - 标准格式: dinov2/dinov3 + _vit_ + {small/base/large} + _{patch_size}
        例: dinov2_vit_small_14 / dinov3_vit_large_14
      - 紧凑格式: dinov2/dinov3 + _vits/vitb/vitl + {patch_size}
        例: dinov3_vitb16 / dinov3_vits16plus / dinov2_vits14
    """
    from anomalib.models.components.dinov2 import vision_transformer as dinov2_models

    # 映射到 DinoV2Loader 的 MODEL_CONFIGS 架构标识
    arch_map = {
        "s": "small",   "small": "small",
        "b": "base",    "base": "base",
        "l": "large",   "large": "large",
    }

    # 先用正则提取: (vit)(s|b|l|small|base|large)(\d*)(plus)?
    import re
    # 匹配 vits16, vitb16, vitl16, vits16plus, vit_small, vit_base 等
    m = re.search(r'vit([sbl]|small|base|large)(\d+)(plus)?', encoder_name, re.IGNORECASE)
    if not m:
        raise ValueError(f"无法从 '{encoder_name}' 中解析出 ViT 架构大小 (s/b/l) 和 patch_size")

    arch_short = m.group(1).lower()
    patch_size = int(m.group(2))
    architecture = arch_map.get(arch_short)
    if not architecture:
        raise ValueError(f"未知的 ViT 架构标识: '{arch_short}'")

    # 查找构造函数
    ctor_name = f"vit_{architecture}"
    ctor = getattr(dinov2_models, ctor_name, None)
    if ctor is None:
        raise ValueError(f"dinov2_models 中没有 {ctor_name} (可用: vit_small, vit_base, vit_large)")

    log.info(f"  手动构建: {ctor_name}(patch_size={patch_size}) [从 '{encoder_name}' 解析]")
    model = ctor(patch_size=patch_size)
    return model


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Anomaly Detection Training (Multi-model)")

    '''        
        # PatchCore（默认，1 epoch 建库）
        python train/train.py

        # SimpleNet（自动 100 epochs 训练）
        python train/train.py --model simplenet

        # RD（自动 100 epochs + input-size 跟随 image-size）
        python train/train.py --model rd

        # PaDiM（1 epoch，最快基线）
        python train/train.py --model padim

        # 自定义图像大小 + epochs
        python train/train.py --model rd --image-size 224 --max-epochs 50

    '''

    # 实验
    parser.add_argument("--exp-name", type=str, default=None,
                        help="实验名称 (默认: train_<时间戳>)")

    # ---- 模型选择 ----
    parser.add_argument("--model", type=str, default="patchcore",
                        choices=["patchcore", "simplenet", "rd", "padim", "anomaly_dino"],
                        help="模型选择: patchcore / simplenet(SimpleNet) / rd(ReverseDistillation) / padim / anomaly_dino(DINOv2)")

    # 数据集
    parser.add_argument("--dataset-root", type=str,default=r"Z:\14-调试数据\lxm\Dataset\Anomalib\TB")
    parser.add_argument("--normal-dir", type=str, default="OK_V2")
    parser.add_argument("--abnormal-dir", type=str, default="defects")
    parser.add_argument("--mask-dir", type=str, default="masks")
    parser.add_argument("--extensions", type=str, nargs="+",default=[".bmp", ".jpg", ".png"])

    # 数据加载
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=16, help="评估批次大小（15GB显存建议≤8，避免阈值计算OOM）")
    parser.add_argument("--num-workers", type=int, default=4,help="数据加载进程数（NAS 盘建议 4~8，SSD 可设更高）")
    parser.add_argument("--pin-memory", action="store_true", default=True,help="锁页内存加速 CPU→GPU 数据传输")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-split-ratio", type=float, default=0.2)
    parser.add_argument("--val-split-ratio", type=float, default=0.5)

    # 模型通用参数
    parser.add_argument("--backbone", type=str, default="resnet18",
                        help="骨干网络名称（各模型有不同默认值，--model 切换时自动适配）")
    parser.add_argument("--layers", type=str, nargs="+",
                        default=["layer2", "layer3"],
                        help="特征提取层（各模型有不同默认值）")
    parser.add_argument("--pre-trained",
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--image-size", type=int, default=256,
                        help="输入图像尺寸 (正方形 H=W)，默认 256")
    parser.add_argument("--input-size", type=int, nargs=2, default=[256, 256],
                        metavar=("H", "W"),
                        help="输入图像尺寸 (H W)，RD 模型必需，默认跟随 --image-size")

    # 模型特有参数
    # PatchCore
    parser.add_argument("--coreset-sampling-ratio", type=float, default=0.1,
                        help="[PatchCore] 核心集采样率。降低此值可大幅减少内存占用 (建议0.01~0.05)")
    # SimpleNet
    parser.add_argument("--perlin-threshold", type=float, default=0.2,
                        help="[SimpleNet] Perlin 噪声阈值")
    # PaDiM
    parser.add_argument("--n-features", type=int, default=None,
                        help="[PaDiM] 降维后保留的特征数 (resnet18=100, wrn50=550)")
    # RD
    parser.add_argument("--anomaly-map-mode", type=str, default="add",
                        choices=["add", "multiply"],
                        help="[RD] 异常图生成模式: add / multiply")
    
    # 权重加载选项 (Timm / MMDet / DINO 三选一)
    weight_group = parser.add_mutually_exclusive_group()
    weight_group.add_argument("--weights-path", type=str, default=r'weights/timm/resnet18.pth',
                        help="[Timm本地] 本地权重路径。放到 weights/timm/ 目录下。")
    weight_group.add_argument("--mmdet-weights", type=str, default=None,
                        help="[MMDetection] MMDetection 权重路径。若指定此项，--backbone 需改为 torchvision 名称 (如 resnet50)")
    weight_group.add_argument("--dino-weights", type=str, default=None,
                        help="[DINOv2/DINOv3] DINO 预训练权重 .pth 路径。配合 --model anomaly_dino 使用。")

    # DINO 特有参数 (anomaly_dino 模型专用)
    parser.add_argument("--encoder-name", type=str, default="dinov2_vit_small_14",
                        help="[AnomalyDINO] DINO 编码器名称 (dinov2_vit_small_14 / dinov2_vit_base_14 / dinov3_vits16 等)")
    parser.add_argument("--masking", action=argparse.BooleanOptionalAction, default=False,
                        help="[AnomalyDINO] 是否启用 PCA 掩码抑制背景特征")
    parser.add_argument("--coreset-subsampling", action=argparse.BooleanOptionalAction, default=True,
                        help="[AnomalyDINO] 是否启用 coreset 降采样减少显存")

    # 阈值
    parser.add_argument("--image-sensitivity", type=float, default=0.5,
                        help="图像级灵敏度 (0~1)")
    parser.add_argument("--pixel-sensitivity", type=float, default=0.7,
                        help="像素级灵敏度 (0~1)，越高 pred_mask 区域越大")

    # 训练
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--max-epochs", type=int, default=1)
    parser.add_argument("--save-last",
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-top-k", type=int, default=0)
    parser.add_argument("--save-weights-only",
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-test", action="store_true",
                        help="跳过测试阶段（显存不足时可用）")
    parser.add_argument("--vis-samples", type=int, default=100,
                        help="[阶段4] 训练后采样可视化数量（默认100张）。设0跳过，完整预测请用 predict.py")
    parser.add_argument("--no-vis-test", action="store_true",
                        help="[阶段3b] 测试时不输出图片到 images/ 目录（只算指标，省磁盘）")

    args = parser.parse_args()

    # 根据 model 自动调整 backbone/layers 默认值
    _apply_model_defaults(args)

    return args


if __name__ == "__main__":
    main()




# python train/train.py 
#   --exp-name train_mmdet_R50_patchcore 
#   --model patchcore 
#   --backbone resnet50 
#   --layers layer2 layer3 
#   --mmdet-weights weights\mmdet\TB_R50_20260520.pth 
#   --coreset-sampling-ratio 0.5 
#   --pixel-sensitivity 0.5 
#   --no-vis-test --vis-samples 100




# python train/train.py 
#   --exp-name train_timm_R18_patchcore 
#   --model patchcore 
#   --backbone resnet18 
#   --layers layer2 layer3 
#   --weights-path weights\timm\resnet18.pth 
#   --coreset-sampling-ratio 0.1 
#   --pixel-sensitivity 0.7 
#   --no-vis-test --vis-samples 100


# python train/train.py 
#   --exp-name train_dinov2_vits14 
#   --model anomaly_dino 
#   --dino-weights weights\dino\dinov2_vits14_pretrain.pth 
#   --encoder-name dinov2_vit_small_14 
#   --masking --no-coreset-subsampling 
#   --image-size 252 
#   --no-vis-test --vis-samples 100



# python train/train.py 
#   --exp-name train_dinov3_vitb16 
#   --model anomaly_dino 
#   --dino-weights Z:\14-调试数据\lxm\Projects\DINOv3\pretrain_models\dinov3\dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth 
#   --encoder-name dinov3_vitb16 
#   --masking --coreset-subsampling 
#   --image-size 252 
#   --no-vis-test --vis-samples 100
