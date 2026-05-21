from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path


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


def main(args: argparse.Namespace | None = None) -> None:
    _add_src_to_path()

    from anomalib.engine import Engine
    from anomalib.models import Patchcore
    from anomalib.post_processing import PostProcessor
    from anomalib.visualization import ImageVisualizer

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

    # ---- 预测数据 ----
    data_path = Path(args.predict_path)
    log.info(f"data_path = {data_path}")

    # ---- 模型 & 后处理 ----
    post_processor = PostProcessor(
        image_sensitivity=args.image_sensitivity,
        pixel_sensitivity=args.pixel_sensitivity,
    )
    log.info(f"pixel_threshold = {1.0 - args.pixel_sensitivity:.2f} (sensitivity={args.pixel_sensitivity})")

    backbone_arg = args.backbone
    if args.mmdet_weights:
        # 如果使用 MMDetection 自定义权重，需要先实例化对应的 torchvision backbone
        log.info(f"使用 MMDetection 自定义主干网络: {args.backbone}")
        import torchvision.models as tv_models
        if hasattr(tv_models, args.backbone):
            backbone_arg = getattr(tv_models, args.backbone)()
        else:
            log.error(f"❌ torchvision 中不支持该模型: {args.backbone}")
            sys.exit(1)
        # 注意：推理阶段不需要加载 .pth 权重文件，因为 engine.predict 会自动从 ckpt 中恢复整个模型的状态！
        # 这里只要把空壳网络塞进去，让 Anomalib 的 TimmFeatureExtractor 能用 torch.fx 抓取对应层即可。

    model = Patchcore(
        backbone=backbone_arg,
        layers=tuple(args.layers),
        pre_trained=args.pre_trained if not args.mmdet_weights else False,
        coreset_sampling_ratio=args.coreset_sampling_ratio,
        visualizer=ImageVisualizer(output_dir=images_dir),
        post_processor=post_processor,
    )

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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tubu Patchcore Prediction")

    # ===== 必要输入 =====
    parser.add_argument("--train-dir","--t", type=str, required=True,
                        help="Training experiment dir (e.g. results/train_20260423_134550), auto-find .ckpt")
    parser.add_argument("--predict-path","--p", type=str, required=True,
                        help="Image or folder to predict")

    # ===== 可调参数 =====
    parser.add_argument("--pixel-sensitivity", type=float, default=0.5,
                        help="Pixel sensitivity (0~1). Higher = larger pred_mask area. Default 0.5")
    parser.add_argument("--image-sensitivity", type=float, default=0.5,
                        help="Image sensitivity (0~1). Default 0.5")

    # 模型结构（需与训练一致，一般不用动）
    parser.add_argument("--backbone", type=str, default="wide_resnet50_2")
    parser.add_argument("--layers", type=str, nargs="+", default=["layer2", "layer3"])
    parser.add_argument("--pre-trained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--coreset-sampling-ratio", type=float, default=0.01)
    
    # 支持 MMDetection 自定义权重推理
    parser.add_argument("--mmdet-weights", action="store_true",
                        help="若训练时使用了 mmdet 权重，请在推理时带上此 Flag，同时确保 --backbone 正确")

    # 硬件
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--devices", type=int, default=1)

    return parser.parse_args()


if __name__ == "__main__":
    main()
