#!/usr/bin/env python3
"""
CLI for manual labeling: interactive OpenCV GUI to draw/correct YOLO bounding boxes.
Supports Stage 1 (dial bbox) and Stage 2 (digit bboxes in ROI crop).
Uses pipeline prediction as optional starting point.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import PipelineReader, ManualLabeler


def main():
    parser = argparse.ArgumentParser(
        description="Manual labeling: draw/correct YOLO boxes with OpenCV GUI"
    )
    parser.add_argument(
        "--src",
        type=str,
        required=True,
        help="Source image directory (or images/ subdir auto-detected)",
    )
    parser.add_argument(
        "--dst",
        type=str,
        required=True,
        help="Output directory; writes dst/images/ and dst/labels/",
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["stage1", "stage2"],
        required=True,
        help="Labeling task: stage1=dial bbox, stage2=digit bboxes in ROI",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip images that already have a label in dst/labels/",
    )
    parser.add_argument(
        "--review",
        action="store_true",
        help="Re-open all previously labeled images for correction",
    )
    parser.add_argument(
        "--no-pipeline-hint",
        action="store_true",
        help="Disable model prediction overlay",
    )
    parser.add_argument(
        "--stage1-labels",
        type=str,
        default=None,
        help="For stage2: path to stage1 labels dir (default: dst.parent/labels)",
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

    args = parser.parse_args()
    workspace = Path(args.workspace or Path.cwd())
    src = Path(args.src)
    dst = Path(args.dst)

    if not src.exists():
        print(f"Error: Source not found: {src}")
        sys.exit(1)

    reader = None
    if not args.no_pipeline_hint:
        try:
            reader = PipelineReader(
                topology_path=args.topology,
                workspace_root=workspace,
                device=args.device,
            )
        except Exception as e:
            print(f"Warning: Could not load PipelineReader: {e}")
            print("Continuing without pipeline hints.")

    stage1_labels_dir = None
    if args.stage == "stage2" and args.stage1_labels:
        stage1_labels_dir = Path(args.stage1_labels)
        if not stage1_labels_dir.exists():
            print(f"Warning: Stage1 labels dir not found: {stage1_labels_dir}")

    labeler = ManualLabeler(stage=args.stage, reader=reader)
    stats = labeler.label_directory(
        src_images=src,
        dst=dst,
        resume=args.resume,
        use_pipeline_hint=not args.no_pipeline_hint,
        review=args.review,
        stage1_labels_dir=stage1_labels_dir,
    )

    print(f"Labeled: {stats['labeled']}, Skipped: {stats['skipped']}")
    if stats["quit"]:
        print("(Session quit early)")


if __name__ == "__main__":
    main()
