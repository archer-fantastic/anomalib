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
    log.info(f"  backbone            : {args.backbone}")
    log.info(f"  layers              : {args.layers}")
    log.info(f"  image_size          : {args.image_size}")
    log.info(f"  max_epochs          : {args.max_epochs} "
             f"({'特征建库' if args.max_epochs == 1 else '梯度训练'})")
    _use_fp16 = args.model in ("simplenet", "rd")
    log.info(f"  precision           : {'FP16 (混合精度)' if _use_fp16 else 'FP32 (Memory Bank 方法无需梯度)'}")
    log.info(f"  vis_samples         : {args.vis_samples} ({'跳过' if args.vis_samples == 0 else '训练后随机采样'})")

    # ---- 预训练权重加载（返回 backbone_arg 或 None）----
    backbone_arg = _load_backbone_weights(args, log)

    # ---- 根据模型类型实例化 ----
    model = _create_model(args, backbone_arg, images_dir, post_processor, log)

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
        "mmdet_weights", "weights_path", "pre_trained",
        "n_features", "coreset_sampling_ratio", "perlin_threshold",
        "anomaly_map_mode", "image_sensitivity", "pixel_sensitivity",
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
    log.info(f"  可视化图: {images_dir}")
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

    n_samples = min(max(int(total * sample_ratio), 1), max_samples)
    sampled = rng.sample(all_images, n_samples)

    log.info(f"  抽样: {n_samples}/{total} 张 "
             f"(ratio={sample_ratio}, max={max_samples})")

    success, fail = 0, 0
    t0 = time.time()
    for i, img_path in enumerate(sampled, 1):
        try:
            engine.predict(
                model=model,
                ckpt_path=str(ckpt_path),
                data_path=str(img_path),
                return_predictions=False,
            )
            success += 1
            if i % 20 == 0 or i == n_samples:
                log.info(f"  进度: {i}/{n_samples} ({success} 成功, {fail} 失败)")
        except Exception as e:
            fail += 1
            log.warning(f"  [{i}/{n_samples}] 失败 {img_path.name}: {e}")

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
}


def _apply_model_defaults(args: argparse.Namespace) -> None:
    """根据 --model 覆盖 backbone/layers/max_epochs/input-size 的默认值（仅当用户未手动指定时生效）。"""
    cfg = _MODEL_DEFAULTS[args.model]
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
    """加载预训练权重，返回 backbone_arg (模型实例 / 字符串 / None)。"""
    backbone_arg = args.backbone

    # ---- RD 模型不支持自定义权重注入（decoder 需要字符串查表）----
    if args.model == "rd":
        if args.mmdet_weights or (args.weights_path and Path(args.weights_path).is_file()):
            log.warning(f"  ⚠ RD 不支持自定义权重，已忽略 --mmdet-weights/--weights-path")
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

    # 模式 C/D: 在线下载 或 随机初始化
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

    use_pretrained = (args.weights_path is None and not args.mmdet_weights) and args.pre_trained
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
        log.error(f"❌ 不支持的模型: {args.model} (可选: patchcore/simplenet/rd/padim)")
        sys.exit(1)


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
                        choices=["patchcore", "simplenet", "rd", "padim"],
                        help="模型选择: patchcore / simplenet(SimpleNet) / rd(ReverseDistillation) / padim")

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
    
    # 权重加载选项 (Timm / MMDet 二选一)
    weight_group = parser.add_mutually_exclusive_group()
    weight_group.add_argument("--weights-path", type=str, default=r'weights/timm/resnet18.pth',
                        help="[Timm本地] 本地权重路径。放到 weights/timm/ 目录下。")
    weight_group.add_argument("--mmdet-weights", type=str, default=None,
                        help="[MMDetection] MMDetection 权重路径。若指定此项，--backbone 需改为 torchvision 名称 (如 resnet50)")

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

    args = parser.parse_args()

    # 根据 model 自动调整 backbone/layers 默认值
    _apply_model_defaults(args)

    return args


if __name__ == "__main__":
    main()
