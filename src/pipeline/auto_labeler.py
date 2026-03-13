"""
Auto-labeler: Use the pipeline to generate YOLO-format labels for new images.
Supports stage1 (dial bbox) and stage2 (digit bboxes in ROI crop).
"""

from pathlib import Path
from typing import Dict, Literal

from .reader import PipelineReader, ReadingResult

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


def _xyxy_to_yolo(cx: float, cy: float, w: float, h: float, img_w: int, img_h: int) -> str:
    """Convert normalized cx,cy,w,h to YOLO label line."""
    return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"


class AutoLabeler:
    """
    Uses PipelineReader to auto-label new images for Stage 1 or Stage 2 training.
    """

    def __init__(self, reader: PipelineReader, conf_thresh: float = 0.7):
        """
        Args:
            reader: PipelineReader instance (models must be loadable)
            conf_thresh: Minimum Stage 1 dial confidence to include an image
        """
        self.reader = reader
        self.conf_thresh = conf_thresh

    def label_directory(
        self,
        src_images: Path,
        dst: Path,
        stage: Literal["stage1", "stage2"],
        dry_run: bool = False,
    ) -> Dict[str, int]:
        """
        Auto-label all images in src_images directory.

        Args:
            src_images: Directory containing images (or path to images/ subdir)
            dst: Output directory; creates dst/images/ and dst/labels/
            stage: 'stage1' = dial bbox per image; 'stage2' = digit bboxes in ROI crop
            dry_run: If True, only count, do not write files

        Returns:
            {"processed": N, "labeled": N, "skipped_low_conf": N}
        """
        if not CV2_AVAILABLE:
            raise RuntimeError("OpenCV required. pip install opencv-python")

        images_dir = src_images / "images" if (src_images / "images").exists() else src_images
        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")

        ext = {".jpg", ".jpeg", ".png", ".bmp"}
        image_paths = sorted(
            p for p in images_dir.iterdir()
            if p.suffix.lower() in ext
        )

        dst_images = dst / "images"
        dst_labels = dst / "labels"
        if not dry_run:
            dst_images.mkdir(parents=True, exist_ok=True)
            dst_labels.mkdir(parents=True, exist_ok=True)

        stats = {"processed": 0, "labeled": 0, "skipped_low_conf": 0}

        for img_path in image_paths:
            stats["processed"] += 1
            try:
                result = self.reader.predict(str(img_path))
            except Exception:
                continue

            if result is None:
                stats["skipped_low_conf"] += 1
                continue

            if result.dial_confidence < self.conf_thresh:
                stats["skipped_low_conf"] += 1
                continue

            stem = img_path.stem

            if stage == "stage1":
                if result.dial_box is None:
                    stats["skipped_low_conf"] += 1
                    continue
                x1, y1, x2, y2 = result.dial_box
                h, w = result.image_shape
                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                label_line = _xyxy_to_yolo(cx, cy, bw, bh, w, h)
                if not dry_run:
                    label_path = dst_labels / f"{stem}.txt"
                    with open(label_path, "w") as f:
                        f.write(label_line)
                    out_img = dst_images / img_path.name
                    if not out_img.exists() or str(out_img.resolve()) != str(img_path.resolve()):
                        import shutil
                        shutil.copy2(img_path, out_img)
                stats["labeled"] += 1

            elif stage == "stage2":
                if result.dial_box is None or not result.digit_boxes_in_roi:
                    continue
                img = self.reader._load_image(str(img_path))
                roi_crop, _ = self.reader._crop_roi(img, result.dial_box)
                roi_h, roi_w = roi_crop.shape[:2]
                label_lines = []
                for box in result.digit_boxes_in_roi:
                    x1, y1, x2, y2 = box
                    cx = ((x1 + x2) / 2) / roi_w
                    cy = ((y1 + y2) / 2) / roi_h
                    bw = (x2 - x1) / roi_w
                    bh = (y2 - y1) / roi_h
                    label_lines.append(_xyxy_to_yolo(cx, cy, bw, bh, roi_w, roi_h))
                if not dry_run:
                    label_path = dst_labels / f"{stem}.txt"
                    with open(label_path, "w") as f:
                        f.writelines(label_lines)
                    out_img = dst_images / f"{stem}{img_path.suffix}"
                    cv2.imwrite(str(out_img), roi_crop)
                stats["labeled"] += 1

        return stats
