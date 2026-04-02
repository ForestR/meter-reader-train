#!/usr/bin/env python3
"""
Test Stage 1 post-processing (DIAL ROI) using ground-truth YOLO segmentation masks.

Reads images and label .txt files (normalized polygon format), rasterizes dial (0) and
decimal_section (1) masks, then runs validate_masks + build_dial_roi from postprocess.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Project root for mega_meter_reader imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2
import numpy as np

from mega_meter_reader.stage1.postprocess import build_dial_roi, validate_masks


def parse_yolo_seg_label_lines(text: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Parse YOLO segment label text into lists of polygon arrays (float32 Nx2 in pixel coords).

    Each non-empty line: class_id x1 y1 x2 y2 ... (normalized 0..1).
    class 0 = dial, class 1 = decimal_section.
    """
    dial_polys: List[np.ndarray] = []
    dec_polys: List[np.ndarray] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 7 or (len(parts) - 1) % 2 != 0:
            continue
        cls = int(float(parts[0]))
        coords = [float(x) for x in parts[1:]]
        n = len(coords) // 2
        poly = np.array(
            [[coords[2 * i], coords[2 * i + 1]] for i in range(n)],
            dtype=np.float64,
        )
        if cls == 0:
            dial_polys.append(poly)
        elif cls == 1:
            dec_polys.append(poly)
    return dial_polys, dec_polys


def polys_to_mask(
    polys: List[np.ndarray],
    width: int,
    height: int,
) -> np.ndarray:
    """
    Rasterize normalized polygons to dense float32 mask (H, W) in [0, 1].
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    for poly in polys:
        if poly.size == 0:
            continue
        pts = poly.copy()
        pts[:, 0] *= width
        pts[:, 1] *= height
        pts_i = np.round(pts).astype(np.int32)
        cv2.fillPoly(mask, [pts_i], 255)
    return (mask.astype(np.float32) / 255.0).clip(0.0, 1.0)


def masks_from_yolo_label_file(
    label_path: Path,
    hw: Tuple[int, int],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[str]]:
    """
    Load YOLO .txt label and produce dial_mask, decimal_mask (H,W) float32 [0,1].

    Returns (None, None, warnings) if file missing/empty or no polygons.
    """
    warnings: List[str] = []
    h, w = hw
    if not label_path.is_file():
        warnings.append(f"Label file not found: {label_path}")
        return None, None, warnings
    text = label_path.read_text(encoding="utf-8", errors="replace")
    dial_polys, dec_polys = parse_yolo_seg_label_lines(text)
    if not dial_polys:
        warnings.append("No dial (class 0) polygon in label.")
    if not dec_polys:
        warnings.append("No decimal_section (class 1) polygon in label.")
    if not dial_polys or not dec_polys:
        return None, None, warnings
    dial = polys_to_mask(dial_polys, w, h)
    dec = polys_to_mask(dec_polys, w, h)
    return dial, dec, warnings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 1 post-process test: GT YOLO masks -> DIAL ROI images"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Dataset root (e.g. data/stage1_dial_seg/meter-panel) with images/<split> and labels/<split>",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Which split to scan (default: val)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="results/test_postprocess_gt",
        help="Output directory for DIAL ROI jpgs",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N images (for smoke test)",
    )
    args = parser.parse_args()

    data_root = Path(args.data).resolve()
    images_dir = data_root / "images" / args.split
    labels_dir = data_root / "labels" / args.split
    out_dir = Path(args.out).resolve()

    if not images_dir.is_dir():
        print(f"Error: images directory not found: {images_dir}", file=sys.stderr)
        sys.exit(1)
    if not labels_dir.is_dir():
        print(f"Error: labels directory not found: {labels_dir}", file=sys.stderr)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = sorted(
        p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in exts
    )
    if args.limit is not None:
        image_paths = image_paths[: max(0, args.limit)]

    n_total = len(image_paths)
    n_valid = 0
    n_invalid = 0
    all_warnings: List[str] = []

    for img_path in image_paths:
        stem = img_path.stem
        label_path = labels_dir / f"{stem}.txt"
        img = cv2.imread(str(img_path))
        if img is None:
            n_invalid += 1
            wmsg = f"{stem}: failed to read image"
            all_warnings.append(wmsg)
            print(f"[skip] {wmsg}")
            continue

        h, w = img.shape[:2]
        dial, dec, w_load = masks_from_yolo_label_file(label_path, (h, w))
        if dial is None or dec is None:
            n_invalid += 1
            for w in w_load:
                all_warnings.append(f"{stem}: {w}")
            print(f"[invalid] {stem}: {'; '.join(w_load)}")
            continue

        ok, w_val = validate_masks(dial, dec)
        if not ok:
            n_invalid += 1
            for w in w_val:
                all_warnings.append(f"{stem}: {w}")
            print(f"[invalid] {stem}: {'; '.join(w_val)}")
            continue

        out = build_dial_roi(img, dial, dec)
        n_valid += 1
        for w in out.warnings:
            all_warnings.append(f"{stem}: {w}")
        out_path = out_dir / f"{stem}.jpg"
        if not cv2.imwrite(str(out_path), out.dial_roi):
            n_invalid += 1
            n_valid -= 1
            print(f"[error] {stem}: failed to write {out_path}")
            continue
        status = "invalid" if out.is_invalid else "ok"
        extra = f" warnings={out.warnings}" if out.warnings else ""
        print(f"[{status}] {stem} -> {out_path}{extra}")

    print()
    print("===== Summary =====")
    print(f"Dataset:     {data_root}")
    print(f"Split:       {args.split}")
    print(f"Output:      {out_dir}")
    print(f"Total:       {n_total}")
    print(f"Valid ROI:   {n_valid}")
    print(f"Invalid:     {n_invalid}")
    if all_warnings:
        print(f"Warning lines: {len(all_warnings)}")
    print("===================")


if __name__ == "__main__":
    main()
