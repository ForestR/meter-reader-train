#!/usr/bin/env python3
"""
Prepare Stage 2 training data: crop images to dial ROI and re-normalize digit boxes.

Uses data/basic_e2e/labels/ (Stage 1 GT dial ROI boxes) to crop each image from
data/basic/images/ to the dial region. Re-normalizes digit coordinates from
data/basic/labels/ into that crop space and relabels all classes to 0 ("digit").

Output: data/basic_stage2/images/ and data/basic_stage2/labels/
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

try:
    from PIL import Image
except ImportError:
    Image = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare Stage 2 data: crop to dial ROI, re-normalize digit boxes, nc=1"
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
        "--src-roi-labels",
        type=str,
        default="data/basic_e2e/labels",
        help="Source dial ROI labels (nc=1, one box per image)",
    )
    parser.add_argument(
        "--dst",
        type=str,
        default="data/basic_stage2",
        help="Destination directory for Stage 2 data",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.05,
        help="Fractional padding around ROI crop (0.05 = 5% on each side)",
    )
    return parser.parse_args()


def read_roi_label(path: Path) -> Optional[Tuple[float, float, float, float]]:
    """Read dial ROI box from e2e label. Returns (cx, cy, w, h) or None if empty."""
    if not path.exists():
        return None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cx = float(parts[1])
                cy = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
                return (cx, cy, w, h)
            except (ValueError, IndexError):
                continue
    return None


def read_digit_labels(path: Path) -> List[Tuple[float, float, float, float]]:
    """Read digit boxes from label file. Returns list of (cx, cy, w, h) in normalized coords."""
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
                cx = float(parts[1])
                cy = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
                boxes.append((cx, cy, w, h))
            except (ValueError, IndexError):
                continue
    return boxes


def norm_to_pixel(cx: float, cy: float, w: float, h: float, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    """Convert normalized YOLO box to pixel corners (x1, y1, x2, y2)."""
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return (x1, y1, x2, y2)


def pixel_to_norm(x1: float, y1: float, x2: float, y2: float, crop_w: float, crop_h: float) -> Tuple[float, float, float, float]:
    """Convert pixel corners to normalized YOLO box (cx, cy, w, h)."""
    cx = (x1 + x2) / 2 / crop_w
    cy = (y1 + y2) / 2 / crop_h
    w = (x2 - x1) / crop_w
    h = (y2 - y1) / crop_h
    return (cx, cy, w, h)


def process_image(
    img_path: Path,
    digit_labels_path: Path,
    roi_labels_path: Path,
    dst_images: Path,
    dst_labels: Path,
    padding: float,
) -> bool:
    """
    Crop image to dial ROI, re-normalize digit boxes, write output.
    Returns True if processed successfully.
    """
    roi = read_roi_label(roi_labels_path)
    if roi is None:
        return False

    digit_boxes = read_digit_labels(digit_labels_path)
    if not digit_boxes:
        return False

    if Image is None:
        print("Error: PIL/Pillow required. Install with: pip install Pillow")
        sys.exit(1)

    img = Image.open(img_path).convert("RGB")
    img_w, img_h = img.size

    cx, cy, w, h = roi
    # Expand ROI by padding
    pad_w = w * padding
    pad_h = h * padding
    x1_n = max(0.0, cx - w / 2 - pad_w)
    y1_n = max(0.0, cy - h / 2 - pad_h)
    x2_n = min(1.0, cx + w / 2 + pad_w)
    y2_n = min(1.0, cy + h / 2 + pad_h)

    x1_px = int(x1_n * img_w)
    y1_px = int(y1_n * img_h)
    x2_px = int(x2_n * img_w)
    y2_px = int(y2_n * img_h)

    crop_w = x2_px - x1_px
    crop_h = y2_px - y1_px
    if crop_w < 1 or crop_h < 1:
        return False

    cropped = img.crop((x1_px, y1_px, x2_px, y2_px))

    out_labels = []
    for (dcx, dcy, dw, dh) in digit_boxes:
        px1, py1, px2, py2 = norm_to_pixel(dcx, dcy, dw, dh, img_w, img_h)
        # Clip to crop region
        clip_x1 = max(px1, x1_px)
        clip_y1 = max(py1, y1_px)
        clip_x2 = min(px2, x2_px)
        clip_y2 = min(py2, y2_px)
        if clip_x2 <= clip_x1 or clip_y2 <= clip_y1:
            continue
        # Convert to crop-relative coords
        rel_x1 = clip_x1 - x1_px
        rel_y1 = clip_y1 - y1_px
        rel_x2 = clip_x2 - x1_px
        rel_y2 = clip_y2 - y1_px
        ncx, ncy, nw, nh = pixel_to_norm(rel_x1, rel_y1, rel_x2, rel_y2, crop_w, crop_h)
        if nw < 0.001 or nh < 0.001:
            continue
        out_labels.append(f"0 {ncx:.6f} {ncy:.6f} {nw:.6f} {nh:.6f}\n")

    if not out_labels:
        return False

    stem = img_path.stem
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    out_img = dst_images / f"{stem}{img_path.suffix}"
    cropped.save(out_img)

    out_label = dst_labels / f"{stem}.txt"
    with open(out_label, "w") as f:
        f.writelines(out_labels)

    return True


def main():
    args = parse_args()
    workspace = Path.cwd()
    src_images = workspace / args.src_images
    src_labels = workspace / args.src_labels
    src_roi_labels = workspace / args.src_roi_labels
    dst = workspace / args.dst

    if not src_images.exists():
        print(f"Error: Source images not found: {src_images}")
        sys.exit(1)
    if not src_labels.exists():
        print(f"Error: Source labels not found: {src_labels}")
        sys.exit(1)
    if not src_roi_labels.exists():
        print(f"Error: Source ROI labels not found: {src_roi_labels}")
        sys.exit(1)

    dst_images = dst / "images"
    dst_labels = dst / "labels"
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    processed = 0
    skipped = 0

    for img_path in sorted(src_images.iterdir()):
        if img_path.suffix.lower() not in image_extensions:
            continue
        stem = img_path.stem
        digit_label_path = src_labels / f"{stem}.txt"
        roi_label_path = src_roi_labels / f"{stem}.txt"
        if process_image(
            img_path, digit_label_path, roi_label_path,
            dst_images, dst_labels, args.padding
        ):
            processed += 1
        else:
            skipped += 1

    print(f"Processed: {processed} images")
    if skipped:
        print(f"Skipped: {skipped} images")
    print(f"Output: {dst_images}, {dst_labels}")


if __name__ == "__main__":
    main()
