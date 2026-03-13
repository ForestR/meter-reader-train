#!/usr/bin/env python3
"""
Convert digit-level YOLO labels to end-to-end meter labels.

Reads label files with multiple digit bounding boxes (class 0-9) and writes
a single class-0 meter_display bounding box per image (union of all digit boxes).
"""

import argparse
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert digit-level labels to single-class meter labels"
    )
    parser.add_argument(
        "--src-labels",
        type=str,
        default="data/basic/labels",
        help="Source labels directory (digit-level)",
    )
    parser.add_argument(
        "--src-images",
        type=str,
        default="data/basic/images",
        help="Source images directory",
    )
    parser.add_argument(
        "--dst",
        type=str,
        default="data/basic_e2e",
        help="Destination directory for converted data",
    )
    parser.add_argument(
        "--no-symlink",
        action="store_true",
        help="Copy images instead of symlinking (default: symlink)",
    )
    return parser.parse_args()


def convert_label_file(src_path: Path, dst_path: Path) -> bool:
    """
    Convert a single digit-level label file to meter-level.
    Returns True if converted, False if skipped (empty source).
    """
    lines = []
    with open(src_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                x1 = cx - w / 2
                x2 = cx + w / 2
                y1 = cy - h / 2
                y2 = cy + h / 2
                lines.append((x1, y1, x2, y2))
            except (ValueError, IndexError):
                continue

    if not lines:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        dst_path.touch()
        return True

    x1_min = min(p[0] for p in lines)
    y1_min = min(p[1] for p in lines)
    x2_max = max(p[2] for p in lines)
    y2_max = max(p[3] for p in lines)

    cx = (x1_min + x2_max) / 2
    cy = (y1_min + y2_max) / 2
    w = x2_max - x1_min
    h = y2_max - y1_min

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_path, "w") as f:
        f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    return True


def main():
    args = parse_args()
    workspace = Path.cwd()
    src_labels = workspace / args.src_labels
    src_images = workspace / args.src_images
    dst = workspace / args.dst

    if not src_labels.exists():
        print(f"Error: Source labels directory not found: {src_labels}")
        sys.exit(1)
    if not src_images.exists():
        print(f"Error: Source images directory not found: {src_images}")
        sys.exit(1)

    dst_labels = dst / "labels"
    dst_images = dst / "images"
    dst_labels.mkdir(parents=True, exist_ok=True)

    if dst_images.exists():
        if dst_images.is_symlink():
            dst_images.unlink()
        elif dst_images.is_dir():
            print(f"Warning: {dst_images} exists and is not a symlink; skipping images setup")
    if not dst_images.exists():
        if args.no_symlink:
            import shutil
            shutil.copytree(src_images, dst_images)
            print(f"Copied images: {src_images} -> {dst_images}")
        else:
            dst_images.symlink_to(src_images.resolve())
            print(f"Symlinked images: {dst_images} -> {src_images}")

    converted = 0
    skipped = 0
    for src_file in sorted(src_labels.glob("*.txt")):
        dst_file = dst_labels / src_file.name
        if convert_label_file(src_file, dst_file):
            converted += 1
        else:
            skipped += 1

    print(f"Converted: {converted} label files")
    if skipped:
        print(f"Skipped: {skipped} files")
    print(f"Output: {dst_labels}")
    print(f"Images: {dst_images}")


if __name__ == "__main__":
    main()
