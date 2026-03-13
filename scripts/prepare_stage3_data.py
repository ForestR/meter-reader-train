#!/usr/bin/env python3
"""
Prepare Stage 3 training data: extract per-digit crops for classification.

Reads data/basic/ images and labels (class 0-9, per-digit boxes). For each digit,
crops the digit bounding box from the image and saves to a classification folder
structure. Applies 80/20 per-class train/val split (deterministic, seed=42).

Output: data/digit_crops/train/{0-9}/ and data/digit_crops/val/{0-9}/
"""

import argparse
import random
import sys
from pathlib import Path
from typing import List, Tuple

try:
    from PIL import Image
except ImportError:
    Image = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare Stage 3 data: extract digit crops for classification (nc=10)"
    )
    parser.add_argument(
        "--src-images",
        type=str,
        default="data/basic/images",
        help="Source images directory",
    )
    parser.add_argument(
        "--src-labels",
        type=str,
        default="data/basic/labels",
        help="Source digit labels directory (nc=10)",
    )
    parser.add_argument(
        "--dst",
        type=str,
        default="data/digit_crops",
        help="Destination directory for digit crops",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.1,
        help="Fractional padding around digit box (0.1 = 10% on each side)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Fraction of data for validation (0.2 = 20%%)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split",
    )
    return parser.parse_args()


def read_digit_boxes(path: Path) -> List[Tuple[int, float, float, float, float]]:
    """
    Read digit boxes from label file.
    Returns list of (class_id, cx, cy, w, h) in normalized coords.
    """
    boxes = []
    if not path.exists():
        return boxes
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cls = int(parts[0])
                cx = float(parts[1])
                cy = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
                if 0 <= cls <= 9:
                    boxes.append((cls, cx, cy, w, h))
            except (ValueError, IndexError):
                continue
    return boxes


def norm_to_pixel(cx: float, cy: float, w: float, h: float, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    """Convert normalized YOLO box to pixel corners (x1, y1, x2, y2)."""
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    return (x1, y1, x2, y2)


def apply_padding(x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int, padding: float) -> Tuple[int, int, int, int]:
    """Expand box by padding fraction, clipped to image bounds."""
    w = x2 - x1
    h = y2 - y1
    pad_w = int(w * padding)
    pad_h = int(h * padding)
    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(img_w, x2 + pad_w)
    y2 = min(img_h, y2 + pad_h)
    return (x1, y1, x2, y2)


def main():
    args = parse_args()
    workspace = Path.cwd()
    src_images = workspace / args.src_images
    src_labels = workspace / args.src_labels
    dst = workspace / args.dst

    if not src_images.exists():
        print(f"Error: Source images not found: {src_images}")
        sys.exit(1)
    if not src_labels.exists():
        print(f"Error: Source labels not found: {src_labels}")
        sys.exit(1)

    if Image is None:
        print("Error: PIL/Pillow required. Install with: pip install Pillow")
        sys.exit(1)

    random.seed(args.seed)

    # Collect all (class_id, img_path, x1, y1, x2, y2) for cropping
    crops_by_class: dict[int, List[Tuple[Path, int, int, int, int]]] = {i: [] for i in range(10)}

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    for img_path in sorted(src_images.iterdir()):
        if img_path.suffix.lower() not in image_extensions:
            continue
        stem = img_path.stem
        label_path = src_labels / f"{stem}.txt"
        boxes = read_digit_boxes(label_path)
        if not boxes:
            continue

        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size

        for cls, cx, cy, w, h in boxes:
            x1, y1, x2, y2 = norm_to_pixel(cx, cy, w, h, img_w, img_h)
            x1, y1, x2, y2 = apply_padding(x1, y1, x2, y2, img_w, img_h, args.padding)
            if x2 <= x1 or y2 <= y1:
                continue
            crop_w = x2 - x1
            crop_h = y2 - y1
            if crop_w < 2 or crop_h < 2:
                continue
            crops_by_class[cls].append((img_path, x1, y1, x2, y2))

    # Per-class train/val split
    train_dst = dst / "train"
    val_dst = dst / "val"
    for cls in range(10):
        (train_dst / str(cls)).mkdir(parents=True, exist_ok=True)
        (val_dst / str(cls)).mkdir(parents=True, exist_ok=True)

    total_train = 0
    total_val = 0
    global_crop_idx = 0

    for cls in range(10):
        items = crops_by_class[cls]
        random.shuffle(items)
        n_val = max(0, int(len(items) * args.val_split))

        for i, (img_path, x1, y1, x2, y2) in enumerate(items):
            split = "val" if i < n_val else "train"
            if split == "val":
                total_val += 1
            else:
                total_train += 1

            img = Image.open(img_path).convert("RGB")
            crop = img.crop((x1, y1, x2, y2))
            out_name = f"{img_path.stem}_{cls}_{global_crop_idx:06d}.png"
            out_path = dst / split / str(cls) / out_name
            crop.save(out_path)
            global_crop_idx += 1

    print(f"Train crops: {total_train}")
    print(f"Val crops: {total_val}")
    print(f"Output: {dst}/train/{{0-9}}/, {dst}/val/{{0-9}}/")


if __name__ == "__main__":
    main()
