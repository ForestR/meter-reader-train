#!/usr/bin/env python3
"""
Demo script for the 3-stage pipeline: image in, reading out.
Supports single image, directory batch, visualization, and auto-labeling.
"""

import argparse
import sys
from pathlib import Path

# Add project root for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import PipelineReader, ReadingResult, AutoLabeler, draw_pipeline_result

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


def main():
    parser = argparse.ArgumentParser(
        description="Demo: run 3-stage pipeline on images (image in, reading out)"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Single image path",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="Directory of images to process",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for --save-vis or --dir",
    )
    parser.add_argument(
        "--save-vis",
        action="store_true",
        help="Save annotated visualization image",
    )
    parser.add_argument(
        "--show-digit-labels",
        action="store_true",
        help="Show class/conf labels on digit bboxes (default: hide)",
    )
    parser.add_argument(
        "--topology",
        type=str,
        default="configs/model_topology.yaml",
        help="Path to model_topology.yaml",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help="Project root (default: current directory)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for inference (e.g. 0, cpu)",
    )
    parser.add_argument(
        "--auto-label",
        type=str,
        choices=["stage1", "stage2"],
        default=None,
        help="Run auto-labeling instead of demo",
    )
    parser.add_argument(
        "--dst",
        type=str,
        default="data/auto_label",
        help="Destination for --auto-label output",
    )
    parser.add_argument(
        "--conf-thresh",
        type=float,
        default=0.7,
        help="Min confidence for auto-labeling (default: 0.7)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="With --auto-label: only count, do not write",
    )

    args = parser.parse_args()
    workspace = Path(args.workspace or Path.cwd())

    if args.auto_label and args.dst:
        # Auto-labeling mode
        src = Path(args.dir or args.image or ".")
        if not src.exists():
            print(f"Error: Source not found: {src}")
            sys.exit(1)
        if src.is_file():
            src = src.parent
        dst = Path(args.dst)
        reader = PipelineReader(
            topology_path=args.topology,
            workspace_root=workspace,
            device=args.device,
        )
        labeler = AutoLabeler(reader, conf_thresh=args.conf_thresh)
        stats = labeler.label_directory(src, dst, stage=args.auto_label, dry_run=args.dry_run)
        print(f"Processed: {stats['processed']}, Labeled: {stats['labeled']}, Skipped (low conf): {stats['skipped_low_conf']}")
        if args.dry_run:
            print("(Dry run - no files written)")
        return

    if args.auto_label and not args.dst:
        print("Error: --auto-label requires --dst")
        sys.exit(1)

    reader = PipelineReader(
        topology_path=args.topology,
        workspace_root=workspace,
        device=args.device,
    )

    if args.image:
        # Single image
        img_path = Path(args.image)
        if not img_path.exists():
            print(f"Error: Image not found: {img_path}")
            sys.exit(1)
        result = reader.predict(str(img_path))
        if result is None:
            print(f"{img_path.name} -> (no dial detected)")
        else:
            print(f"{img_path.name} -> {result.reading}  (conf: {result.confidence:.2f})")
        if args.save_vis and result is not None and CV2_AVAILABLE:
            img = cv2.imread(str(img_path))
            vis = draw_pipeline_result(img, result, show_digit_labels=args.show_digit_labels)
            out_dir = Path(args.output or img_path.parent)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{img_path.stem}_vis{img_path.suffix}"
            cv2.imwrite(str(out_path), vis)
            print(f"Saved visualization: {out_path}")

    elif args.dir:
        # Directory batch
        dir_path = Path(args.dir)
        if not dir_path.exists():
            print(f"Error: Directory not found: {dir_path}")
            sys.exit(1)
        images_dir = dir_path / "images" if (dir_path / "images").exists() else dir_path
        ext = {".jpg", ".jpeg", ".png", ".bmp"}
        image_paths = sorted(
            p for p in images_dir.iterdir()
            if p.suffix.lower() in ext
        )
        out_dir = Path(args.output or dir_path) if args.save_vis else None
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        for img_path in image_paths:
            result = reader.predict(str(img_path))
            if result is None:
                reading = "(no dial)"
                conf = 0.0
            else:
                reading = result.reading
                conf = result.confidence
            rows.append((img_path.name, reading, conf))
            print(f"{img_path.name} -> {reading}  (conf: {conf:.2f})")
            if args.save_vis and result is not None and CV2_AVAILABLE:
                img = cv2.imread(str(img_path))
                vis = draw_pipeline_result(img, result, show_digit_labels=args.show_digit_labels)
                out_path = out_dir / f"{img_path.stem}_vis{img_path.suffix}"
                cv2.imwrite(str(out_path), vis)

        print()
        print(f"Processed {len(rows)} images")
        if out_dir and args.save_vis:
            print(f"Visualizations saved to: {out_dir}")

    else:
        print("Specify --image or --dir")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
