"""
Dataset Health Check & Repair Tool for Anomalib Folder Dataset.

Features:
  1. Check image-mask correspondence (one-to-one matching)
  2. Move orphan images/masks to backup directory
  3. (Optional) Generate missing masks from LabelMe JSON annotations

Usage:
  # Check only (dry-run, no file operations)
  python dataset_check.py --root D:/dataset/TB

  # Check + move unmatched files to backup
  python dataset_check.py --root D:/dataset/TB --action move

  # Generate masks from LabelMe JSON first, then check
  python dataset_check.py --root D:/dataset/TB --generate-masks

  # Full repair: generate masks + move orphans
  python dataset_check.py --root D:/dataset/TB --generate-masks --action move
"""

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXTS = {".bmp", ".jpg", ".jpeg", ".png"}


def list_images(root: Path) -> list[Path]:
    """Recursively collect all image files."""
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in EXTS
    )


def safe_print(msg: str) -> None:
    """Print safely on Windows console (avoid UnicodeEncodeError)."""
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode("ascii"))


# ---------------------------------------------------------------------------
# LabelMe JSON -> Mask conversion
# ---------------------------------------------------------------------------

def parse_labelme_json(json_path: Path) -> np.ndarray | None:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    height = data["imageHeight"]
    width = data["imageWidth"]
    shapes = data.get("shapes", [])
    if not shapes:
        return None

    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    for shape in shapes:
        if shape["shape_type"] == "polygon":
            points = [tuple(p) for p in shape["points"]]
            if len(points) >= 3:
                draw.polygon(points, fill=255)
        elif shape["shape_type"] == "rectangle":
            x0, y0 = shape["points"][0]
            x1, y1 = shape["points"][1]
            draw.rectangle([min(x0, y1), min(y0, y1), max(x0, x1), max(y0, y1)], fill=255)

    return np.array(mask)


def generate_masks(labelme_root: Path, mask_root: Path) -> dict:
    """Generate binary masks from LabelMe JSON files.
    Returns stats dict with keys: converted, skipped, errors.
    """
    labelme_root = Path(labelme_root)
    mask_root = Path(mask_root)

    json_files = sorted(labelme_root.rglob("*.json"))
    safe_print(f"\n[Generate Masks] Found {len(json_files)} JSON files")

    stats = {"converted": 0, "skipped": 0, "errors": 0}

    for jpath in tqdm(json_files, desc="Converting", unit="file"):
        try:
            mask = parse_labelme_json(jpath)
            if mask is None or mask.sum() == 0:
                stats["skipped"] += 1
                continue

            rel = jpath.relative_to(labelme_root).with_suffix(".png")
            out_path = mask_root / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)

            ok = cv2.imwrite(str(out_path), mask)
            if not ok:
                Image.fromarray(mask).save(str(out_path))
            stats["converted"] += 1
        except Exception as e:
            stats["errors"] += 1

    safe_print(f"  Converted: {stats['converted']} | Skipped: {stats['skipped']} | Errors: {stats['errors']}")
    return stats


# ---------------------------------------------------------------------------
# Image-Mask correspondence check
# ---------------------------------------------------------------------------

def check_correspondence(image_dir: Path, mask_dir: Path) -> dict:
    """
    Check one-to-one correspondence between images and masks.

    Matching rule: image stem == mask stem (same relative path, ignore extension).

    Returns dict with:
      - total_images, total_masks
      - missing_mask_images: list of image Paths without matching mask
      - extra_masks: list of mask Paths without matching image
      - matched_count
    """
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)

    all_images = list_images(image_dir)
    all_masks = list_images(mask_dir)

    # Build lookup: (relative path without extension) -> Path
    img_lookup: dict[str, Path] = {}
    for p in all_images:
        rel = p.relative_to(image_dir)
        key = str(rel.with_suffix(""))
        img_lookup[key] = p

    mask_lookup: dict[str, Path] = {}
    for p in all_masks:
        rel = p.relative_to(mask_dir)
        key = str(rel.with_suffix(""))
        mask_lookup[key] = p

    missing_mask_images = []
    extra_masks = []

    for key, ipath in img_lookup.items():
        if key not in mask_lookup:
            missing_mask_images.append(ipath)

    for key, mpath in mask_lookup.items():
        if key not in img_lookup:
            extra_masks.append(mpath)

    matched = len(all_images) - len(missing_mask_images)

    return {
        "total_images": len(all_images),
        "total_masks": len(all_masks),
        "missing_mask_images": missing_mask_images,
        "extra_masks": extra_masks,
        "matched_count": matched,
    }


# ---------------------------------------------------------------------------
# Move orphan files to backup
# ---------------------------------------------------------------------------

def move_orphans(result: dict, root: Path, dry_run: bool = False) -> None:
    """Move unmatched images and extra masks to _backup folder under root."""
    root = Path(root)
    backup = root / "_backup"
    moved_images = result.get("missing_mask_images", [])
    moved_masks = result.get("extra_masks", [])

    if not moved_images and not moved_masks:
        safe_print("\nAll good! No orphan files.")
        return

    if dry_run:
        safe_print(f"\n[DRY RUN] Would move {len(moved_images)} images + {len(moved_masks)} masks to {backup}")
        return

    backup.mkdir(exist_ok=True)
    n_img = 0
    n_msk = 0

    for src in moved_images:
        rel = src.relative_to(root)  # relative to dataset root
        dst = backup / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        n_img += 1

    for src in moved_masks:
        rel = src.relative_to(root)
        dst = backup / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        n_msk += 1

    safe_print(f"\nMoved {n_img} orphan images + {n_msk} extra masks -> {backup}")
    safe_print("You can delete the backup folder after confirming everything works.")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(result: dict) -> None:
    r = result
    safe_print("=" * 60)
    safe_print("DATASET HEALTH REPORT")
    safe_print("=" * 60)
    safe_print(f"  Total images : {r['total_images']}")
    safe_print(f"  Total masks  : {r['total_masks']}")
    safe_print(f"  Matched pairs: {r['matched_count']}")
    safe_print(f"  Missing mask : {len(r['missing_mask_images'])} images have NO mask")
    safe_print(f"  Extra masks  : {len(r['extra_masks'])} masks have NO image")

    if r["missing_mask_images"]:
        safe_print("\n  Images without mask:")
        dirs = [str(p.parent.name) for p in r["missing_mask_images"]]
        for d, cnt in Counter(dirs).most_common(10):
            safe_print(f"    {d}: {cnt}")

    if r["extra_masks"]:
        safe_print("\n  Extra masks without image:")
        dirs = [str(p.parent.name) for p in r["extra_masks"]]
        for d, cnt in Counter(dirs).most_common(10):
            safe_print(f"    {d}: {cnt}")

    safe_print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset Health Check & Repair Tool")
    parser.add_argument("--root", "-r", required=True,
                        help="Dataset root containing OK/, defects/, masks/")
    parser.add_argument("--normal-dir", default="OK",
                        help="Normal images subfolder (default: OK)")
    parser.add_argument("--abnormal-dir", default="defects",
                        help="Abnormal images subfolder (default: defects)")
    parser.add_argument("--mask-dir", default="masks",
                        help="Mask subfolder (default: masks)")

    parser.add_argument("--action", choices=["check", "move"], default="check",
                        help="check=only report; move=also move orphans to _backup/")
    parser.add_argument("--generate-masks", action="store_true",
                        help="Generate missing masks from LabelMe JSON annotations first")

    args = parser.parse_args()

    root = Path(args.root)
    abnormal_dir = root / args.abnormal_dir
    mask_dir = root / args.mask_dir

    safe_print(f"Dataset root: {root}")

    # Step 1: Optionally generate masks from LabelMe JSON
    if args.generate_masks:
        # Look for JSON alongside images
        generate_masks(abnormal_dir, mask_dir)

    # Step 2: Check correspondence
    result = check_correspondence(abnormal_dir, mask_dir)
    print_report(result)

    # Step 3: Move orphans if requested
    if args.action == "move":
        move_orphans(result, root, dry_run=False)

    # Summary verdict
    n_bad = len(result["missing_mask_images"]) + len(result["extra_masks"])
    if n_bad == 0:
        safe_print("\n[PASS] All images and masks are perfectly paired.")
    else:
        safe_print(f"\n[ISSUE] {n_bad} orphan files found. Use --action move to clean up.")


if __name__ == "__main__":
    main()
