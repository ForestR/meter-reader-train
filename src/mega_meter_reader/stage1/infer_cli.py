"""CLI: run Stage 1 segmentation + DIAL ROI on an image."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None  # type: ignore[misc, assignment]

from mega_meter_reader.stage1.predict import run_stage1


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="Stage 1 inference: DIAL ROI from segment model")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained *.pt weights")
    parser.add_argument("--source", type=str, required=True, help="Input image path")
    parser.add_argument("--out", type=str, required=True, help="Output image path for DIAL_ROI")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args(argv)

    if YOLO is None:
        print("ultralytics is required: pip install ultralytics", file=sys.stderr)
        sys.exit(1)

    src = Path(args.source)
    if not src.is_file():
        print(f"Not found: {src}", file=sys.stderr)
        sys.exit(1)

    img = cv2.imread(str(src))
    if img is None:
        print(f"Failed to read image: {src}", file=sys.stderr)
        sys.exit(1)

    model = YOLO(args.weights)
    out, _results = run_stage1(
        model,
        img,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), out.dial_roi)

    if out.is_invalid:
        print("WARNING: is_invalid=True", file=sys.stderr)
        for w in out.warnings:
            print(f"  {w}", file=sys.stderr)
    else:
        for w in out.warnings:
            print(w, file=sys.stderr)


if __name__ == "__main__":
    main()
