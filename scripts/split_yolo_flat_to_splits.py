#!/usr/bin/env python3
"""
Split flat YOLO layout (dst/images/*.jpg + dst/labels/*.txt) into
images/{train,val,test} and labels/{train,val,test}, then write data.yaml.

Pairs by stem; skips images without labels unless --allow-missing-labels.
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore


_ROOT = Path(__file__).resolve().parent.parent


def _collect_pairs(
    src: Path,
    allow_missing_labels: bool,
) -> List[Tuple[Path, Path]]:
    """Return list of (image_path, label_path). Label may be missing if allowed (use placeholder)."""
    img_dir = src / "images"
    lbl_dir = src / "labels"
    if not img_dir.is_dir() or not lbl_dir.is_dir():
        raise SystemExit(f"Expected {src}/images and {src}/labels directories")

    ext = {".jpg", ".jpeg", ".png", ".bmp"}
    pairs: List[Tuple[Path, Path]] = []
    for img_path in sorted(img_dir.iterdir()):
        if not img_path.is_file() or img_path.suffix.lower() not in ext:
            continue
        lp = lbl_dir / f"{img_path.stem}.txt"
        if not lp.is_file():
            if allow_missing_labels:
                pairs.append((img_path, lp))  # caller may copy empty
            else:
                print(f"[skip] no label: {img_path.name}", file=sys.stderr)
            continue
        pairs.append((img_path, lp))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split flat YOLO images/labels into train/val/test subfolders + data.yaml"
    )
    parser.add_argument(
        "--src",
        type=str,
        required=True,
        help="Labeled set root with flat images/ and labels/",
    )
    parser.add_argument(
        "--dst",
        type=str,
        required=True,
        help="Output dataset root (created)",
    )
    parser.add_argument(
        "--train",
        type=float,
        default=0.7,
        help="Fraction for train (default 0.7)",
    )
    parser.add_argument(
        "--val",
        type=float,
        default=0.15,
        help="Fraction for val (default 0.15); remainder -> test",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Shuffle seed",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copy",
    )
    parser.add_argument(
        "--allow-missing-labels",
        action="store_true",
        help="Include images even if .txt missing (writes empty label)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Dataset name in data.yaml (default: dst folder name)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print split counts only",
    )
    args = parser.parse_args()

    if args.train <= 0 or args.val < 0 or args.train + args.val >= 1.0:
        print("Error: require train > 0, val >= 0, train + val < 1", file=sys.stderr)
        sys.exit(1)

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    pairs = _collect_pairs(src, args.allow_missing_labels)
    if not pairs:
        print("No image/label pairs found", file=sys.stderr)
        sys.exit(1)

    random.seed(args.seed)
    order = list(range(len(pairs)))
    random.shuffle(order)
    n = len(order)
    n_train = int(n * args.train)
    n_val = int(n * args.val)
    n_test = n - n_train - n_val

    splits: List[Tuple[str, List[int]]] = [
        ("train", order[:n_train]),
        ("val", order[n_train : n_train + n_val]),
        ("test", order[n_train + n_val :]),
    ]

    print(f"total={n} train={len(splits[0][1])} val={len(splits[1][1])} test={len(splits[2][1])}")

    if args.dry_run:
        return

    op = shutil.move if args.move else shutil.copy2

    for split_name, indices in splits:
        idir = dst / "images" / split_name
        ldir = dst / "labels" / split_name
        idir.mkdir(parents=True, exist_ok=True)
        ldir.mkdir(parents=True, exist_ok=True)
        for i in indices:
            img_path, lbl_path = pairs[i]
            op(img_path, idir / img_path.name)
            if lbl_path.is_file():
                op(lbl_path, ldir / lbl_path.name)
            else:
                (ldir / f"{img_path.stem}.txt").write_text("", encoding="utf-8")

    data_yaml = {
        "path": str(dst),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 1,
        "names": ["digit"],
    }
    if args.name:
        data_yaml = {"name": args.name, **data_yaml}
    if yaml is not None:
        with open(dst / "data.yaml", "w", encoding="utf-8") as f:
            yaml.dump(
                data_yaml,
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )
    else:
        # minimal yaml without PyYAML
        lines = [
            f"path: {dst}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "nc: 1",
            "names:",
            "  0: digit",
        ]
        (dst / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {dst / 'data.yaml'}")


if __name__ == "__main__":
    main()
