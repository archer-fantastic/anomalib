"""
将 LabelMe JSON 标注文件转换为二值 mask 图片。
用于 anomalib 的 Folder 数据集分割模式。

用法:
    python convert_labelme_to_mask.py --input Z:/14-调试数据/lxm/Dataset/Anomalib/涂布/漏金属 --output Z:/14-调试数据/lxm/Dataset/Anomalib/涂布/masks
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm


def parse_labelme_json(json_path: Path) -> np.ndarray | None:
    """解析单个 LabelMe JSON 文件，返回二值 mask。"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    height = data["imageHeight"]
    width = data["imageWidth"]
    shapes = data.get("shapes", [])

    if not shapes:
        return None

    # 创建空白 mask (白色背景)
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    for shape in shapes:
        if shape["shape_type"] == "polygon":
            points = [tuple(p) for p in shape["points"]]
            if len(points) >= 3:
                draw.polygon(points, fill=255)
        elif shape["shape_type"] == "rectangle":
            points = shape["points"]
            x0, y0 = points[0]
            x1, y1 = points[1]
            draw.rectangle([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)], fill=255)

    return np.array(mask)


def main():
    parser = argparse.ArgumentParser(description="LabelMe JSON -> 二值 Mask")
    parser.add_argument("--input", "-i", required=True, help="包含 LabelMe JSON 的根目录（递归搜索）")
    parser.add_argument("--output", "-o", required=True, help="输出 mask 的根目录（保持子目录结构）")
    args = parser.parse_args()

    input_root = Path(args.input)
    output_root = Path(args.output)

    # 递归查找所有 JSON 文件
    json_files = sorted(input_root.rglob("*.json"))
    print(f"找到 {len(json_files)} 个 LabelMe JSON 文件")

    converted = 0
    skipped = 0
    errors = 0

    for json_path in tqdm(json_files, desc="转换中", unit="文件"):
        try:
            mask = parse_labelme_json(json_path)
            if mask is None or mask.sum() == 0:
                skipped += 1
                continue

            # 计算相对路径，保持目录结构
            rel_path = json_path.relative_to(input_root)
            # 将 .json 后缀改为 .png
            mask_rel_path = rel_path.with_suffix(".png")
            output_path = output_root / mask_rel_path
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # 保存为 PNG 格式的二值 mask
            output_path_str = str(output_path)
            success = cv2.imwrite(output_path_str, mask)
            if not success:
                # cv2 失败时改用 PIL 保存
                Image.fromarray(mask).save(output_path_str)
            converted += 1

            if converted <= 3:
                print(f"  [示例] {json_path.name} -> {mask_rel_path}")

        except Exception as e:
            print(f"错误: {json_path} - {e}")
            errors += 1

    print(f"\n转换完成！")
    print(f"  转换成功: {converted}")
    print(f"  跳过(无标注): {skipped}")
    print(f"  错误: {errors}")
    print(f"  输出目录: {output_root}")

    # 验证输出
    actual_count = len(list(output_root.rglob("*.png")))
    print(f"  实际输出文件数: {actual_count}")
    if actual_count == 0 and converted > 0:
        print("  [!] 警告: 文件未成功写出，请检查磁盘权限或路径!")


if __name__ == "__main__":
    main()
