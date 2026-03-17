#!/usr/bin/env python3
"""
Evaluation script for the 3-stage pipeline.
Parses GT from filenames (value_GT_suffix.ext), runs inference, and reports
accuracy with failure attribution by stage (dial ROI, digit detection, classification).
"""

import argparse
import csv
import sys
from enum import Enum
from pathlib import Path

# Add project root for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import PipelineReader, ReadingResult, draw_pipeline_result

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class Verdict(Enum):
    CORRECT = "correct"
    STAGE1_FAIL = "stage1_fail"
    STAGE2_FAIL = "stage2_fail"
    STAGE3_FAIL = "stage3_fail"
    SKIPPED = "skipped"


def parse_gt(filename: str) -> str | None:
    """
    Extract GT reading from filename.
    Supports: value_GT_suffix.ext and id_N_value_GT_suffix.ext
    Returns None if 'value_' not in filename (non-standard naming).
    """
    stem = Path(filename).stem
    if "value_" not in stem:
        return None
    return stem.split("value_")[1].split("_")[0]


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate pipeline on test images with GT in filenames"
    )
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="Directory of images to evaluate (or dir/images if exists)",
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
        "--output-csv",
        type=str,
        default=None,
        help="Write per-image results to CSV",
    )
    parser.add_argument(
        "--save-failures-vis",
        type=str,
        default=None,
        help="Save visualization only for failed images (Stage 2/3)",
    )
    parser.add_argument(
        "--show-digit-labels",
        action="store_true",
        help="Show class/conf labels on digit bboxes in failure vis",
    )

    args = parser.parse_args()
    workspace = Path(args.workspace or Path.cwd())

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

    reader = PipelineReader(
        topology_path=args.topology,
        workspace_root=workspace,
        device=args.device,
    )

    rows: list[tuple[str, str | None, str, float, Verdict, ReadingResult | None]] = []
    for img_path in image_paths:
        gt = parse_gt(img_path.name)
        if gt is None:
            rows.append((img_path.name, None, "", 0.0, Verdict.SKIPPED, None))
            continue

        result = reader.predict(str(img_path))
        if result is None:
            pred = "(no dial)"
            conf = 0.0
            verdict = Verdict.STAGE1_FAIL
            stored_result = None
        elif len(result.reading) != len(gt):
            pred = result.reading
            conf = result.confidence
            verdict = Verdict.STAGE2_FAIL
            stored_result = result
        elif result.reading != gt:
            pred = result.reading
            conf = result.confidence
            verdict = Verdict.STAGE3_FAIL
            stored_result = result
        else:
            pred = result.reading
            conf = result.confidence
            verdict = Verdict.CORRECT
            stored_result = None

        rows.append((img_path.name, gt, pred, conf, verdict, stored_result))

    # Per-image table
    max_name = max(len(r[0]) for r in rows) if rows else 20
    max_gt = max(len(r[1] or "-") for r in rows) if rows else 10
    for filename, gt, pred, conf, verdict, _ in rows:
        gt_str = gt or "-"
        pred_str = pred if pred else "-"
        v_str = verdict.value
        print(f"{filename:<{max_name}}  GT:{gt_str:<{max_gt}}  pred:{pred_str:<12}  conf:{conf:.2f}  {v_str}")

    # Summary
    with_gt = [r for r in rows if r[4] != Verdict.SKIPPED]
    n_total = len(with_gt)
    n_correct = sum(1 for r in with_gt if r[4] == Verdict.CORRECT)
    n_s1 = sum(1 for r in with_gt if r[4] == Verdict.STAGE1_FAIL)
    n_s2 = sum(1 for r in with_gt if r[4] == Verdict.STAGE2_FAIL)
    n_s3 = sum(1 for r in with_gt if r[4] == Verdict.STAGE3_FAIL)
    n_skipped = sum(1 for r in rows if r[4] == Verdict.SKIPPED)

    pct = lambda a, b: (100.0 * a / b) if b else 0.0
    print()
    print("===== Evaluation Summary =====")
    print(f"Total images (with GT):  {n_total}")
    print(f"  Correct:               {n_correct:>4}  ({pct(n_correct, n_total):.1f}%)")
    print(f"  Stage 1 failures:      {n_s1:>4}  ({pct(n_s1, n_total):.1f}%)  [no dial detected]")
    print(f"  Stage 2 failures:      {n_s2:>4}  ({pct(n_s2, n_total):.1f}%)  [wrong digit count]")
    print(f"  Stage 3 failures:      {n_s3:>4}  ({pct(n_s3, n_total):.1f}%)  [char classification error]")
    print(f"Skipped (no GT in name): {n_skipped}")
    print("==============================")

    # CSV
    if args.output_csv:
        out_csv = Path(args.output_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["filename", "gt", "predicted", "conf", "verdict"])
            for filename, gt, pred, conf, verdict, _ in rows:
                w.writerow([filename, gt or "", pred, f"{conf:.4f}", verdict.value])
        print(f"CSV saved to: {out_csv}")

    # Failure visualizations (Stage 2 and 3 only; Stage 1 has no result to draw)
    if args.save_failures_vis and CV2_AVAILABLE:
        fail_dir = Path(args.save_failures_vis)
        fail_dir.mkdir(parents=True, exist_ok=True)
        saved = 0
        for filename, gt, pred, conf, verdict, stored_result in rows:
            if verdict not in (Verdict.STAGE2_FAIL, Verdict.STAGE3_FAIL) or stored_result is None:
                continue
            img_path = images_dir / filename
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            vis = draw_pipeline_result(img, stored_result, show_digit_labels=args.show_digit_labels)
            out_path = fail_dir / f"{Path(filename).stem}_vis{Path(filename).suffix}"
            cv2.imwrite(str(out_path), vis)
            saved += 1
        print(f"Failure visualizations saved to: {fail_dir} ({saved} images)")


if __name__ == "__main__":
    main()
