"""多模型对比预测脚本。

对同一组图片，用多个训练好的模型分别推理，然后将同一张图各模型的
输出竖着拼接成一张对比图。

实现策略：对每个 train-dir 先完整跑一遍 predict（和 predict.py 完全一致），
所有模型跑完后用 PIL 竖着拼接同名输出图。

用法:
    python train/predict_multi.py ^
        --train-dir results/exp1 results/exp2 ^
        --predict-path Z:/.../defects
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm


def _add_src_to_path() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ══════════════════════════════════════════════════════════════
# 复用 predict.py 的模型创建逻辑（100% 相同）
# ══════════════════════════════════════════════════════════════


def _timm_layer_to_idx(backbone: str, layers: list[str]) -> list[int]:
    import timm
    m = timm.create_model(backbone, pretrained=False, features_only=True, exportable=True)
    names = [i["module"] for i in m.feature_info.info]
    idx = [names.index(L) for L in layers]
    del m
    return idx


def _find_ckpt(train_dir: Path) -> Path:
    ckpt = train_dir / "weights" / "lightning" / "last.ckpt"
    if ckpt.is_file():
        return ckpt
    cand = sorted((train_dir / "weights" / "lightning").glob("*.ckpt"))
    if cand:
        return cand[-1]
    raise FileNotFoundError(f"No checkpoint: {train_dir}")


def _load_dino_encoder(name: str, wp: Path, ver: str):
    from anomalib.models.components.dinov2 import DinoV2Loader
    loader = DinoV2Loader()
    if ver == "dinov3":
        import sys as _sys
        dinov3_root = Path(r"Z:\14-调试数据\lxm\Projects\DINOv3")
        if str(dinov3_root) not in _sys.path:
            _sys.path.insert(0, str(dinov3_root))
        from dinov3.hub import backbones as hub_module
        fn = getattr(hub_module, name)
        return fn(pretrained=False, weights=str(wp))
    mt, arch, ps = loader._parse_name(name)
    try:
        enc = loader.create_model(mt, arch, ps)
    except (ValueError, KeyError):
        from anomalib.models.components.dinov2 import vision_transformer as v
        arch_map = {"s": "small", "b": "base", "l": "large"}
        m = re.search(r'vit([sbl]|small|base|large)(\d+)', name)
        a, p = arch_map[m.group(1).lower()], int(m.group(2))
        enc = getattr(v, f"vit_{a}")(patch_size=p)
    sd = torch.load(wp, map_location="cpu")
    enc.load_state_dict(sd, strict=False)
    return enc


def _get_backbone(cfg: dict):
    """返回 backbone_arg（字符串 或 nn.Module），与 predict.py 完全一致。"""
    model_type = cfg.get("model", "patchcore")
    backbone = cfg.get("backbone", "resnet18")
    layers = cfg.get("layers", ["layer2", "layer3"])
    mmdet_weights = cfg.get("mmdet_weights")
    weights_path = cfg.get("weights_path")
    pre_trained = cfg.get("pre_trained", True)

    # AnomalyDINO
    if model_type == "anomaly_dino":
        return None  # dino 模型内部处理

    # RD / SimpleNet 不支持自定义权重注入
    if model_type in ("rd", "simplenet"):
        return backbone

    # MMDetection 权重注入 — 只建空壳，checkpoint 已有权重（无需 torch.load mmdet 文件）
    if mmdet_weights:
        import torchvision.models as tv_models
        bb = getattr(tv_models, backbone, None)()
        print(f"  ✅ mmdet 架构: {backbone} (checkpoint 会恢复权重)")
        return bb

    # Timm 本地权重
    if weights_path:
        wp = Path(weights_path)
        if wp.is_file():
            import timm
            idx = _timm_layer_to_idx(backbone, layers)
            bb = timm.create_model(backbone, pretrained=False, features_only=True,
                                   exportable=True, out_indices=idx)
            sd = torch.load(wp, map_location="cpu", weights_only=True)
            bb.load_state_dict(sd, strict=False)
            print(f"  ✅ timm 权重: {len(sd)} tensors")
            return bb

    # 默认：字符串（timm 在线下载）
    if pre_trained:
        print(f"  预训练: timm 在线 ({backbone})")
    return backbone


def _create_model_for_predict(train_dir: Path, images_dir: Path):
    """用和 predict.py 完全相同的逻辑创建模型。"""
    _add_src_to_path()

    cfg = json.loads((train_dir / "train_config.json").read_text(encoding="utf-8"))
    model_type = cfg["model"]

    from anomalib.post_processing import PostProcessor
    from anomalib.visualization.image.visualizer import ImageVisualizer

    ps = cfg.get("pixel_sensitivity", 0.5)
    post_processor = PostProcessor(image_sensitivity=cfg.get("image_sensitivity", 0.5), pixel_sensitivity=ps)
    visualizer = ImageVisualizer(output_dir=images_dir)

    if model_type == "anomaly_dino":
        return _create_dino_model(cfg, visualizer, post_processor)

    backbone_arg = _get_backbone(cfg)
    layers = cfg.get("layers", ["layer2", "layer3"])
    pre_trained = cfg.get("pre_trained", True)
    use_pretrained = (not cfg.get("mmdet_weights") and not cfg.get("dino_weights")) and pre_trained

    if model_type == "patchcore":
        from anomalib.models.image.patchcore.lightning_model import Patchcore
        return Patchcore(backbone=backbone_arg, layers=tuple(layers), pre_trained=use_pretrained,
                         coreset_sampling_ratio=cfg.get("coreset_sampling_ratio", 0.1),
                         visualizer=visualizer, post_processor=post_processor)
    elif model_type == "padim":
        from anomalib.models.image.padim.lightning_model import Padim
        return Padim(backbone=backbone_arg, layers=layers, pre_trained=use_pretrained,
                     n_features=cfg.get("n_features"), visualizer=visualizer, post_processor=post_processor)
    elif model_type == "simplenet":
        from anomalib.models.image.supersimplenet.lightning_model import Supersimplenet
        return Supersimplenet(backbone=backbone_arg, layers=layers,
                              perlin_threshold=cfg.get("perlin_threshold", 0.2),
                              visualizer=visualizer, post_processor=post_processor)
    elif model_type == "rd":
        from anomalib.models.image.reverse_distillation.lightning_model import ReverseDistillation
        return ReverseDistillation(backbone=backbone_arg, layers=tuple(layers), pre_trained=use_pretrained,
                                   anomaly_map_mode=cfg.get("anomaly_map_mode", "add"),
                                   visualizer=visualizer, post_processor=post_processor)
    else:
        print(f"❌ 不支持的模型: {model_type}")
        sys.exit(1)


def _create_dino_model(cfg, visualizer, post_processor):
    from anomalib.models.image.anomaly_dino.lightning_model import AnomalyDINO
    from anomalib.models.components.dinov2 import DinoV2Loader
    dino_weights = cfg.get("dino_weights")
    if dino_weights:
        wp = Path(dino_weights)
        ver = "dinov3" if "dinov3" in wp.name.lower() else "dinov2"
        fe = _load_dino_encoder(cfg["encoder_name"], wp, ver)
        _orig = DinoV2Loader.load
        DinoV2Loader.load = lambda s, n: fe
        try:
            model = AnomalyDINO(num_neighbours=1, encoder_name=cfg["encoder_name"],
                                masking=cfg.get("masking", False),
                                coreset_subsample=cfg.get("coreset_subsampling", True),
                                sampling_ratio=cfg.get("coreset_sampling_ratio", 0.1),
                                visualizer=visualizer, post_processor=post_processor)
        finally:
            DinoV2Loader.load = _orig
    else:
        model = AnomalyDINO(num_neighbours=1, encoder_name=cfg["encoder_name"],
                            masking=cfg.get("masking", False),
                            coreset_subsample=cfg.get("coreset_subsampling", True),
                            sampling_ratio=cfg.get("coreset_sampling_ratio", 0.1),
                            visualizer=visualizer, post_processor=post_processor)
    return model


# ══════════════════════════════════════════════════════════════
# 拼接逻辑
# ══════════════════════════════════════════════════════════════


def stitch_vertically(image_paths: list[Path], output_path: Path):
    """竖着拼接多张图片，模型目录名作为标签。"""
    imgs = [Image.open(p) for p in image_paths]
    w = max(i.width for i in imgs)
    h = sum(i.height for i in imgs)

    label_w = 80
    combined = Image.new("RGB", (w + label_w, h), color=(30, 30, 30))

    from PIL import ImageDraw
    draw = ImageDraw.Draw(combined)

    y = 0
    for img, path in zip(imgs, image_paths):
        # 标签 = 模型目录名（path 是 output_root/{model}/images/cat/file.jpg）
        label = path.parent.parent.parent.name[:20]  # ../../model_name
        draw.rectangle([0, y, label_w, y + img.height], fill=(50, 50, 50))
        draw.text((4, y + 6), label, fill="white")
        combined.paste(img, (label_w, y))
        draw.line([(0, y + img.height - 1), (combined.width, y + img.height - 1)],
                  fill=(80, 80, 80), width=1)
        y += img.height

    combined.save(output_path)


# ══════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════


def main():
    _add_src_to_path()
    from anomalib.engine import Engine

    args = _parse_args()
    train_dirs = [Path(d) for d in args.train_dirs]
    predict_path = Path(args.predict_path)
    output_root = Path(args.output)

    output_root.mkdir(parents=True, exist_ok=True)

    print(f"输入图片: {predict_path}")
    print(f"模型数量: {len(train_dirs)}")
    print(f"输出目录: {output_root.resolve()}")
    print(f"{'─'*50}")

    # ── 1. 对每个模型，完整跑一遍 predict ──
    model_output_dirs: dict[str, Path] = {}  # {模型名: 输出images目录}

    for td in train_dirs:
        cfg = json.loads((td / "train_config.json").read_text(encoding="utf-8"))
        short = td.name[:50]
        model_images_dir = output_root / short / "images"

        print(f"\n── {short} ──")
        ckpt = _find_ckpt(td)
        print(f"  checkpoint: {ckpt}")
        print(f"  模型类型  : {cfg['model']}")

        if not model_images_dir.exists() or not any(model_images_dir.iterdir()):
            model = _create_model_for_predict(td, model_images_dir)
            engine = Engine(accelerator="gpu", devices=1, default_root_dir=output_root)
            engine.predict(model=model, ckpt_path=str(ckpt), data_path=str(predict_path))
            print(f"  ✅ 预测完成 → {model_images_dir}")
        else:
            print(f"  ⏭️ 已有结果，跳过")

        model_output_dirs[short] = model_images_dir

    # ── 2. 拼接：找到所有模型共有的图片，竖着拼 ──
    print(f"\n{'─'*50}")
    print("拼接对比图...")

    # 收集：{img_stem: {model_name: image_path}}
    all_models = list(model_output_dirs.keys())

    # 先收集每个模型的所有 stem
    model_stems: list[set[str]] = []
    for short, img_dir in model_output_dirs.items():
        stems = set()
        for p in img_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                stems.add(p.stem)
        model_stems.append(stems)
        print(f"  {short}: {len(stems)} 张")

    # 取交集
    common_stems = model_stems[0]
    for ms in model_stems[1:]:
        common_stems &= ms
    print(f"  共有 {len(common_stems)} 张")

    # 为快速查找，建立 {model_name: {stem: path}}
    stem_to_paths: dict[str, dict[str, Path]] = {}
    for short, img_dir in model_output_dirs.items():
        for p in img_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                if p.stem in common_stems:
                    stem_to_paths.setdefault(p.stem, {})[short] = p

    stitch_dir = output_root / "compare"
    stitch_dir.mkdir(exist_ok=True)

    for stem in tqdm(sorted(common_stems), desc="拼接", unit="img"):
        paths = [stem_to_paths[stem][s] for s in all_models]
        stitch_vertically(paths, stitch_dir / f"{stem}.png")

    print(f"\n✅ 完成！对比图: {stitch_dir.resolve()}")


def _parse_args():
    parser = argparse.ArgumentParser(description="多模型对比预测")
    parser.add_argument("--train-dir", "--t", nargs="+", required=True, dest="train_dirs",
                        help="多个训练目录")
    parser.add_argument("--predict-path", "--p", required=True,
                        help="图片或文件夹")
    parser.add_argument("--output", "-o", default="results/compare",
                        help="输出目录")
    return parser.parse_args()


if __name__ == "__main__":
    main()
