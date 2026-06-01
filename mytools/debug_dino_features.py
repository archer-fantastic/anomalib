"""
DINOv2 特征空间诊断工具

功能:
  1. 加载 DINOv2 编码器 + 自定义权重
  2. 处理一张测试图，打印 grid / patch 信息
  3. PCA 投影可视化 patch 特征的空间连续性
  4. 检查 token 顺序是否与 grid 匹配

用法:
  python mytools/debug_dino_features.py
  python mytools/debug_dino_features.py --image <某张缺陷图路径>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.decomposition import PCA

# 添加 src 到 path
import sys
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))


def main() -> None:
    args = _parse_args()

    # ---- 1. 加载 DINOv2 编码器 ----
    print("=" * 60)
    print("  加载 DINOv2 编码器")
    print("=" * 60)

    from anomalib.models.components.dinov2 import DinoV2Loader

    loader = DinoV2Loader()
    model_type, architecture, patch_size = loader._parse_name(args.encoder_name)
    print(f"  encoder_name : {args.encoder_name}")
    print(f"  model_type   : {model_type}")
    print(f"  architecture : {architecture}")
    print(f"  patch_size   : {patch_size}")

    encoder = loader.create_model(model_type, architecture, patch_size)

    # 加载自定义权重
    weight_path = Path(args.dino_weights)
    if weight_path.is_file():
        print(f"  加载权重     : {weight_path}")
        sd = torch.load(weight_path, map_location="cpu", weights_only=True)
        missing, unexpected = encoder.load_state_dict(sd, strict=False)
        if missing:
            print(f"    缺失键: {len(missing)}")
        if unexpected:
            print(f"    多余键: {len(unexpected)}")
        matched = len(sd) - len(missing)
        print(f"    匹配 Tensor: {matched}/{len(sd)}")
    else:
        print(f"  ⚠ 找不到权重: {weight_path}，将使用随机初始化")

    encoder.eval()

    # ---- 2. 准备测试图像 ----
    print()
    print("=" * 60)
    print("  测试图像信息")
    print("=" * 60)

    if args.image:
        img_path = Path(args.image)
        if not img_path.is_file():
            print(f"  ❌ 文件不存在: {img_path}")
            sys.exit(1)
        # 用 OpenCV 读取，保持原始尺寸
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"  ❌ 无法读取图片: {img_path}")
            sys.exit(1)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img_rgb.shape[:2]
        print(f"  原始尺寸     : {orig_w}x{orig_h}")
    else:
        # 用一张 OK 图作为默认测试
        default_ok = Path(r"Z:\14-调试数据\lxm\Dataset\Anomalib\TB\OK_V2")
        if default_ok.is_dir():
            ok_files = sorted(default_ok.glob("*.*"))
            if ok_files:
                img_path = ok_files[0]
                img_bgr = cv2.imread(str(img_path))
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                print(f"  使用默认 OK 图: {img_path.name}  ({img_rgb.shape[1]}x{img_rgb.shape[0]})")
            else:
                print("  ⚠ 默认目录无图片，使用随机噪声")
                img_rgb = np.random.randint(0, 255, (args.image_size, args.image_size, 3), dtype=np.uint8)
        else:
            print(f"  ⚠ 默认目录不存在: {default_ok}，使用随机噪声")
            print(f"     请指定 --image 参数使用真实图片")
            img_rgb = np.random.randint(0, 255, (args.image_size, args.image_size, 3), dtype=np.uint8)

    # ---- 3. 预处理（AnomalyDINO 的标准预处理）----
    from torchvision.transforms.v2 import Compose, Normalize, Resize, InterpolationMode

    transform = Compose([
        Resize(
            (args.image_size, args.image_size),
            antialias=True,
            interpolation=InterpolationMode.BICUBIC,
        ),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 转为 tensor
    x = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    x = transform(x).unsqueeze(0)  # (1, 3, H, W)

    b, c, h, w = x.shape
    print(f"  预处理后     : {h}x{w}")

    # ---- 4. 检查整除性 ----
    crop_h = h % patch_size
    crop_w = w % patch_size
    pad_top = crop_h // 2
    pad_bottom = crop_h - pad_top
    pad_left = crop_w // 2
    pad_right = crop_w - pad_left
    cropped_h = h - crop_h
    cropped_w = w - crop_w
    grid_h = cropped_h // patch_size
    grid_w = cropped_w // patch_size

    print(f"  整除情况     : crop=({crop_h},{crop_w}) grid=({grid_h},{grid_w})")
    print(f"  总 patch 数  : {grid_h * grid_w}")

    if crop_h > 0 or crop_w > 0:
        x = x[:, :, pad_top : h - pad_bottom, pad_left : w - pad_right]

    # ---- 5. 提取特征 ----
    print()
    print("=" * 60)
    print("  特征提取")
    print("=" * 60)

    # 测试两种模式: reshape=False (原版, 有问题) vs reshape=True (修复后)
    for use_reshape in [False, True]:
        mode_label = "reshape=True (FIXED)" if use_reshape else "reshape=False (ORIGINAL)"
        with torch.inference_mode():
            raw = encoder.get_intermediate_layers(x, n=1, reshape=use_reshape)[0]
        if use_reshape:
            # (B, D, H, W) → flatten to (B, N, D)
            features = raw.flatten(2).transpose(1, 2)
            print(f"  [{mode_label}] raw shape={list(raw.shape)} → features shape={list(features.shape)}")
        else:
            features = raw
            print(f"  [{mode_label}] features shape={list(features.shape)}")

    print(f"  特征 shape   : {list(features.shape)}")
    b_feat, n_feat, d_feat = features.shape
    print(f"  预期 patch   : {grid_h * grid_w}")
    print(f"  实际 patch   : {n_feat}")
    print(f"  特征维度     : {d_feat}")

    if n_feat != grid_h * grid_w:
        print(f"  ⚠️ 数量不匹配！可能 token 顺序或 cls/register token 未正确剥离")
    else:
        print(f"  ✅ patch 数量匹配")

    # ---- 6. PCA 投影可视化 ----
    print()
    print("=" * 60)
    print("  PCA 空间连续性检查")
    print("=" * 60)

    feats_np = features[0].cpu().numpy()  # (N, D)
    pca = PCA(n_components=1)
    feat_1d = pca.fit_transform(feats_np)  # (N, 1)
    # 归一化到 0~255
    fmin, fmax = feat_1d.min(), feat_1d.max()
    feat_norm = ((feat_1d - fmin) / (fmax - fmin + 1e-8) * 255).astype(np.uint8)

    # reshape 回 grid
    grid_img = feat_norm.reshape(grid_h, grid_w)
    print(f"  PCA 投影 grid: {grid_h}x{grid_w}")

    # 水平差分（检查行内连续性）
    diff_h = np.abs(np.diff(grid_img, axis=1)).mean()
    # 垂直差分（检查列内连续性）
    diff_v = np.abs(np.diff(grid_img, axis=0)).mean()
    print(f"  平均行内跳变 : {diff_h:.2f}  (越小越连续)")
    print(f"  平均列内跳变 : {diff_v:.2f}  (越小越连续)")

    if diff_h > 30 or diff_v > 30:
        print("  ⚠️ 跳变较大，特征可能存在空间错乱")
    else:
        print("  ✅ 特征空间连续")

    # 放大显示
    vis_size = 512
    grid_big = cv2.resize(grid_img, (vis_size, vis_size), interpolation=cv2.INTER_NEAREST)
    grid_color = cv2.applyColorMap(grid_big, cv2.COLORMAP_JET)

    args.output.mkdir(parents=True, exist_ok=True)
    save_path = args.output / "dino_pca_grid.png"
    cv2.imwrite(str(save_path), grid_color)
    print(f"  保存 PCA 图 : {save_path}")

    # ---- 7. 渐变图测试（检测 token 顺序是否正确）----
    print()
    print("=" * 60)
    print("  渐变图测试（直接判断 token 顺序）")
    print("=" * 60)

    # 测试多个尺寸，包括非正方形
    test_sizes = [252, 224, 196, 168, 140] if args.image_size == 252 else [args.image_size]

    for test_sz in test_sizes:
        _run_gradient_test(encoder, transform, patch_size, test_sz, args.output)

    # ---- 8. 单像素脉冲测试（最精确的 token 位置映射）----
    print()
    print("=" * 60)
    print("  单像素脉冲测试（精确定位每个 token 对应的图像位置）")
    print("=" * 60)
    _run_pulse_test(encoder, transform, patch_size, args.image_size, args.output)


def _run_gradient_test(encoder, transform, patch_size, img_size, output_dir):
    """用水平/垂直渐变图测试 token 空间顺序。"""
    from torchvision.transforms.v2 import Compose, Normalize, Resize, InterpolationMode

    # 为当前尺寸创建专用 transform（不能用全局的，它会 resize 到 args.image_size）
    local_transform = Compose([
        Resize(
            (img_size, img_size),
            antialias=True,
            interpolation=InterpolationMode.BICUBIC,
        ),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 水平渐变: 左黑→右白
    grad_h = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    for col in range(img_size):
        val = int(col / max(img_size - 1, 1) * 255)
        grad_h[:, col] = [val, val, val]

    # 垂直渐变: 上黑→下白
    grad_v = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    for row in range(img_size):
        val = int(row / max(img_size - 1, 1) * 255)
        grad_v[row, :] = [val, val, val]

    crop_h = img_size % patch_size
    crop_w = img_size % patch_size
    pad_top = crop_h // 2
    pad_bottom = crop_h - pad_top
    pad_left = crop_w // 2
    pad_right = crop_w - pad_left

    for name, grad in [("h_grad", grad_h), ("v_grad", grad_v)]:
        x_g = torch.from_numpy(grad).permute(2, 0, 1).float() / 255.0
        x_g = local_transform(x_g).unsqueeze(0)
        if crop_h > 0 or crop_w > 0:
            x_g = x_g[:, :, pad_top : img_size - pad_bottom, pad_left : img_size - pad_right]

        # 对比两种模式
        for use_reshape in [False, True]:
            with torch.inference_mode():
                raw = encoder.get_intermediate_layers(x_g, n=1, reshape=use_reshape)[0]
            if use_reshape:
                feats = raw.flatten(2).transpose(1, 2)  # (B,D,H,W) → (B,N,D)
            else:
                feats = raw

            pca = PCA(n_components=1)
            proj = pca.fit_transform(feats[0].cpu().numpy())
            norm = ((proj - proj.min()) / (proj.max() - proj.min() + 1e-8) * 255).astype(np.uint8)

            ch = (img_size - crop_h) // patch_size
            cw = (img_size - crop_w) // patch_size
            grid = norm.reshape(ch, cw)

            big = cv2.resize(grid, (512, 512), interpolation=cv2.INTER_NEAREST)
            mode_tag = "fixed" if use_reshape else "orig"
            fname = f"grad_{name}_{img_size}_{mode_tag}.jpg"
            cv2.imwrite(str(output_dir / fname), cv2.applyColorMap(big, cv2.COLORMAP_JET))

            left_m = grid[:, :cw//2].mean()
            right_m = grid[:, cw//2:].mean()
            top_m = grid[:ch//2, :].mean()
            bottom_m = grid[ch//2:, :].mean()
            tag = "✅FIX" if use_reshape else "❌OLD"
            if name.startswith("h"):
                h_ok = "✅" if right_m > left_m else "❌"
                print(f"  [{tag}] {fname} 左={left_m:.0f} 右={right_m:.0f} H:{h_ok}")
            else:
                v_ok = "✅" if bottom_m > top_m else "❌"
                print(f"  [{tag}] {fname} 上={top_m:.0f} 下={bottom_m:.0f} V:{v_ok}")


def _run_pulse_test(encoder, transform, patch_size, img_size, output_dir):
    """在每个 grid 位置放置一个白色脉冲点，看哪个 token 响应最强。"""
    from torchvision.transforms.v2 import Compose, Normalize, Resize, InterpolationMode

    local_transform = Compose([
        Resize(
            (img_size, img_size),
            antialias=True,
            interpolation=InterpolationMode.BICUBIC,
        ),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # 创建黑色背景 + 在特定位置放白点的图像
    crop_h = img_size % patch_size
    crop_w = img_size % patch_size
    ch = (img_size - crop_h) // patch_size
    cw = (img_size - crop_w) // patch_size
    total = ch * cw

    print(f"  图像尺寸: {img_size}, Grid: {ch}x{cw} = {total} patches")

    # 只测试几个关键位置避免太慢
    test_positions = [
        (0, 0, "左上"),
        (0, cw - 1, "右上"),
        (ch - 1, 0, "左下"),
        (ch - 1, cw - 1, "右下"),
        (ch // 2, cw // 2, "中心"),
    ]

    # 收集所有响应
    response_map = np.zeros((total,), dtype=np.float32)

    for row, col, label in test_positions:
        img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        # 在该 patch 中心画一个白色方块
        cy = (row + 0.5) * patch_size + (crop_h // 2 if crop_h else 0)
        cx = (col + 0.5) * patch_size + (crop_w // 2 if crop_w else 0)
        r = patch_size // 3
        y0, y1 = int(cy - r), int(cy + r)
        x0, x1 = int(cx - r), int(cx + r)
        img[y0:y1, x0:x1] = 255

        x_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        x_t = local_transform(x_t).unsqueeze(0)

        pad_top_t = crop_h // 2
        pad_left_t = crop_w // 2
        if crop_h > 0 or crop_w > 0:
            x_t = x_t[:, :, pad_top_t : img_size - (crop_h - pad_top_t), pad_left_t : img_size - (crop_w - pad_left_t)]

        with torch.inference_mode():
            feats = encoder.get_intermediate_layers(x_t, n=1)[0]  # (1, N, D)

        # 计算每个 token 的 L2 范数作为响应强度
        norms = feats[0].norm(dim=-1).cpu().numpy()  # (N,)
        response_map += norms

        # 找到响应最强的 token
        max_idx = norms.argmax()
        max_r, max_c = divmod(max_idx, cw)
        print(f"  脉冲@({label}) grid_pos=({row},{col}) → 最强响应 token_idx={max_idx} grid_pos=({max_r},{max_c})")
        if max_r != row or max_c != col:
            print(f"    ⚠️ 位置不匹配！期望 ({row},{col}) 实际 ({max_r},{max_c})")

    # 可视化响应热力图（综合所有脉冲）
    resp_norm = ((response_map - response_map.min()) / (response_map.max() - response_map.min() + 1e-8) * 255).astype(np.uint8)
    resp_grid = resp_norm.reshape(ch, cw)
    big = cv2.resize(resp_grid, (512, 512), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(output_dir / "pulse_response.jpg"), cv2.applyColorMap(big, cv2.COLORMAP_JET))
    print(f"  保存脉冲响应图: {output_dir / 'pulse_response.jpg'}")


def _check_gradient(grid_map: np.ndarray, label: str, n_rows: int, n_cols: int) -> None:
    """检查渐变图是否从左到右渐变。"""
    # 计算左半边和右半边的均值
    col_mid = n_cols // 2
    left_mean = grid_map[:, :col_mid].mean()
    right_mean = grid_map[:, col_mid:].mean()
    is_correct = right_mean > left_mean

    print(f"    [{label}] 左半均值={left_mean:.1f}, 右半均值={right_mean:.1f} "
          f"{'→ ✅ 正确' if is_correct else '→ ❌ 方向错误'}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DINOv2 特征空间诊断工具")
    parser.add_argument("--dino-weights", type=str,
                        default="weights/dino/dinov2_vits14_pretrain.pth",
                        help="DINOv2 权重路径")
    parser.add_argument("--encoder-name", type=str, default="dinov2_vit_small_14",
                        help="DINO 编码器名称")
    parser.add_argument("--image-size", type=int, default=252,
                        help="输入图像尺寸")
    parser.add_argument("--image", type=str, default=None,
                        help="测试图片路径（默认用随机噪声）")
    parser.add_argument("--output", type=Path, default=Path("results/dino_debug"),
                        help="输出目录")
    return parser.parse_args()


if __name__ == "__main__":
    main()
