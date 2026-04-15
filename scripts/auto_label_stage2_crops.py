#!/usr/bin/env python3
"""
Auto-label pre-cropped dial ROI images with the Stage 2 digit detector only (nc=1).

Does NOT run Stage 1: each image is treated as the full ROI. Writes flat
dst/images/ + dst/labels/ (YOLO detect format, one class 0 per box).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import List

import yaml

# Project root
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import cv2
    from ultralytics import YOLO
except ImportError as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)


def _xyxy_to_yolo_line(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> str:
    cx = ((x1 + x2) / 2) / w
    cy = ((y1 + y2) / 2) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"


def _default_weights(workspace: Path, topology_path: Path) -> Path:
    with open(topology_path) as f:
        topo = yaml.safe_load(f)
    rel = topo["CHECKPOINTS"]["PIPELINE"]["STAGE_2_DIGIT"]
    p = Path(rel)
    if not p.is_absolute():
        p = workspace / p
    return p


def _default_digit_conf(runtime_path: Path) -> float:
    if not runtime_path.is_file():
        return 0.6
    with open(runtime_path) as f:
        rt = yaml.safe_load(f)
    return float(rt.get("CONFIDENCE_THRESHOLDS", {}).get("DIGIT", 0.6))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 2 digit detection only: YOLO labels for pre-cropped dial images"
    )
    parser.add_argument(
        "--src",
        type=str,
        required=True,
        help="Directory of images (flat, or images/ subfolder)",
    )
    parser.add_argument(
        "--dst",
        type=str,
        required=True,
        help="Output root with images/ and labels/ (flat, no train/val yet)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Stage 2 .pt weights (default: STAGE_2_DIGIT from --topology)",
    )
    parser.add_argument(
        "--topology",
        type=str,
        default="configs/model_topology.yaml",
        help="YAML with CHECKPOINTS.PIPELINE.STAGE_2_DIGIT",
    )
    parser.add_argument(
        "--runtime",
        type=str,
        default="configs/runtime_policy.yaml",
        help="YAML with CONFIDENCE_THRESHOLDS.DIGIT (default conf)",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help="Project root (default: repo root)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Ultralytics device, e.g. 0 or cpu",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Min box confidence (default: from runtime_policy DIGIT)",
    )
    parser.add_argument(
        "--skip-empty",
        action="store_true",
        help="Skip writing image/label when no digit detected (default: write empty .txt)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print counts, no files written",
    )
    args = parser.parse_args()

    workspace = Path(args.workspace or _ROOT).resolve()
    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    topology_path = workspace / args.topology
    runtime_path = workspace / args.runtime

    images_dir = src / "images" if (src / "images").is_dir() else src
    if not images_dir.is_dir():
        print(f"Error: not a directory: {images_dir}", file=sys.stderr)
        sys.exit(1)

    weights = Path(args.weights).resolve() if args.weights else _default_weights(workspace, topology_path)
    if not weights.is_file():
        print(f"Error: weights not found: {weights}", file=sys.stderr)
        sys.exit(1)

    conf = args.conf if args.conf is not None else _default_digit_conf(runtime_path)

    ext = {".jpg", ".jpeg", ".png", ".bmp"}
    paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in ext)
    if not paths:
        print(f"No images in {images_dir}", file=sys.stderr)
        sys.exit(1)

    dst_img = dst / "images"
    dst_lbl = dst / "labels"
    if not args.dry_run:
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)

    model = None if args.dry_run else YOLO(str(weights))
    dev = args.device

    n_written = 0
    n_empty = 0
    n_skipped_empty = 0

    for img_path in paths:
        if args.dry_run:
            n_written += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[skip] unreadable: {img_path.name}", file=sys.stderr)
            continue
        h, w = img.shape[:2]

        results = model.predict(img, conf=conf, verbose=False, device=dev)
        lines: List[str] = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                for i in range(len(boxes)):
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = map(float, xyxy)
                    lines.append(_xyxy_to_yolo_line(x1, y1, x2, y2, w, h))

        if not lines and args.skip_empty:
            n_skipped_empty += 1
            continue

        stem = img_path.stem
        out_img = dst_img / img_path.name
        out_lbl = dst_lbl / f"{stem}.txt"
        shutil.copy2(img_path, out_img)
        with open(out_lbl, "w", encoding="utf-8") as f:
            f.writelines(lines)
        n_written += 1
        if not lines:
            n_empty += 1

    print(
        f"images={len(paths)} written={n_written} empty_labels={n_empty} "
        f"skipped_no_detection={n_skipped_empty} dst={dst}"
    )
    if args.dry_run:
        print("(dry-run)")


if __name__ == "__main__":
    main()
