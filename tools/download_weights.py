"""从 timm 下载预训练权重并保存到项目 weights/ 目录。

用法:
    python tools/download_weights.py                    # 默认下载 wide_resnet50_2
    python tools/download_weights.py --model resnet18   # 指定其他 backbone
    python tools/download_weights.py --list              # 列出可用模型

保存位置: weights/<模型名>.pth
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# 项目路径
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
import timm


def download_and_save(model_name: str, output_dir: Path) -> None:
    """下载 timm 预训练权重并保存为 .pth 文件。"""
    print(f"{'=' * 50}")
    print(f"  模型: {model_name}")
    print(f"{'=' * 50}")

    # 1. 创建空骨架 + 在线加载预训练权重
    print(f"\n[1/3] 从 timm 创建模型并下载预训练权重...")
    t0 = time.time()
    model = timm.create_model(
        model_name,
        pretrained=True,
        features_only=True,   # 与 Patchcore 一致
        exportable=True,
        out_indices=(2, 3),   # layer2, layer3（默认值）
    )
    elapsed = time.time() - t0
    print(f"      ✅ 下载完成 ({elapsed:.1f}s)")

    # 2. 统计信息
    state_dict = model.state_dict()
    n_params = sum(v.numel() for v in state_dict.values())
    size_mb = sum(v.numel() * v.element_size() for v in state_dict.values()) / (1024 ** 2)
    print(f"\n[2/3] 权重统计:")
    print(f"      参数量 : {n_params:,} ({n_params / 1e6:.1f}M)")
    print(f"      文件大小: {size_mb:.1f} MB")
    print(f"      层数   : {len(state_dict)} 个 tensor")

    # 3. 保存到 weights/
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"{model_name}.pth"

    print(f"\n[3/3] 保存到: {save_path.resolve()}")
    torch.save(state_dict, save_path)
    print(f"      ✅ 保存成功！\n")

    return save_path


def list_available() -> None:
    """列出常用的 Patchcore 兼容 backbone。"""
    common_models = [
        ("resnet18", "轻量，速度快"),
        ("resnet34", "中等"),
        ("resnet50", "经典"),
        ("wide_resnet50_2", "默认推荐 ✅"),
        ("wide_resnet101_2", "更大更准"),
    ]

    print("\n常用 Patchcore backbone:")
    print("-" * 60)
    for name, desc in common_models:
        try:
            m = timm.create_model(name, pretrained=False, features_only=True)
            n = sum(p.numel() for p in m.parameters())
            del m
            print(f"  {name:<25s} {n / 1e6:>8.1f}M 参数  # {desc}")
        except Exception:
            print(f"  {name:<25s} {'不可用':>20s}       # {desc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="下载 timm 预训练权重到 weights/")
    parser.add_argument("--model", type=str, default="wide_resnet50_2",
                        help="timm 模型名称 (默认: wide_resnet50_2)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="输出目录 (默认: weights/)")
    parser.add_argument("--list", action="store_true",
                        help="列出可用的常用模型")

    args = parser.parse_args()

    if args.list:
        list_available()
        return

    output_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "weights"
    result = download_and_save(args.model, output_dir)

    print("=" * 50)
    print("完成！使用方式:")
    print(f'  python train/train.py --weights-path {result}')
    print("=" * 50)


if __name__ == "__main__":
    main()
