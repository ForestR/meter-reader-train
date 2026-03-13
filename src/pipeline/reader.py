"""
Pipeline Reader: Assembles Stage 1 (dial), Stage 2 (digit), Stage 3 (classify) into
a single inference module. Image in, reading out.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import yaml

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


@dataclass
class ReadingResult:
    """Result of pipeline inference on a single image."""

    reading: str
    confidence: float  # mean of per-digit classification confidences
    dial_confidence: float  # Stage 1 dial detection confidence
    dial_box: Optional[Tuple[float, float, float, float]]  # (x1, y1, x2, y2) in original image pixels
    digit_boxes_in_roi: List[Tuple[float, float, float, float]]  # per-digit in ROI crop pixels
    digit_boxes_in_img: List[Tuple[float, float, float, float]]  # per-digit in original image pixels
    digit_classes: List[int]
    digit_confidences: List[float]
    image_shape: Tuple[int, int]  # (h, w) of original input


class PipelineReader:
    """
    Assembles the 3-stage pipeline: dial detection -> digit detection -> digit classification.
    Loads models lazily on first predict.
    """

    ROI_PADDING = 0.05
    STAGE3_IMGSZ = 80

    def __init__(
        self,
        topology_path: str = "configs/model_topology.yaml",
        runtime_policy_path: str = "configs/runtime_policy.yaml",
        workspace_root: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ):
        """
        Args:
            topology_path: Path to model_topology.yaml
            runtime_policy_path: Path to runtime_policy.yaml
            workspace_root: Project root for resolving relative checkpoint paths
            device: Override device (e.g. '0', 'cpu')
        """
        self.workspace_root = Path(workspace_root or Path.cwd())
        self.topology_path = self.workspace_root / topology_path
        self.runtime_policy_path = self.workspace_root / runtime_policy_path

        with open(self.topology_path) as f:
            self.topology = yaml.safe_load(f)
        with open(self.runtime_policy_path) as f:
            self.runtime = yaml.safe_load(f)

        _dev = device or self.runtime.get("DEVICE", "auto")
        if _dev == "auto":
            try:
                import torch
                self.device = "0" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = _dev
        self.conf_dial = self.runtime.get("CONFIDENCE_THRESHOLDS", {}).get("DIAL", 0.8)
        self.conf_digit = self.runtime.get("CONFIDENCE_THRESHOLDS", {}).get("DIGIT", 0.6)

        self._model_stage1 = None
        self._model_stage2 = None
        self._model_stage3 = None

    def _resolve_checkpoint(self, key: str) -> Path:
        path = self.topology["CHECKPOINTS"]["PIPELINE"][key]
        p = Path(path)
        if not p.is_absolute():
            p = self.workspace_root / p
        return p

    def _ensure_models(self):
        if not ULTRALYTICS_AVAILABLE:
            raise RuntimeError("Ultralytics not installed. pip install ultralytics")
        if self._model_stage1 is None:
            p1 = self._resolve_checkpoint("STAGE_1_DIAL")
            self._model_stage1 = YOLO(str(p1))
        if self._model_stage2 is None:
            p2 = self._resolve_checkpoint("STAGE_2_DIGIT")
            self._model_stage2 = YOLO(str(p2))
        if self._model_stage3 is None:
            p3 = self._resolve_checkpoint("STAGE_3_CLS")
            self._model_stage3 = YOLO(str(p3))

    def _load_image(self, image: Union[str, Path, np.ndarray]) -> np.ndarray:
        if isinstance(image, np.ndarray):
            return image
        path = Path(image)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        if not CV2_AVAILABLE:
            raise RuntimeError("OpenCV required for image loading. pip install opencv-python")
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Failed to load image: {path}")
        return img

    def _run_stage1(self, image: np.ndarray) -> Optional[Tuple[Tuple[float, float, float, float], float]]:
        """Run dial detection. Returns ((x1,y1,x2,y2), conf) or None."""
        self._ensure_models()
        results = self._model_stage1.predict(
            image,
            conf=self.conf_dial,
            verbose=False,
            device=self.device,
        )
        if not results or len(results) == 0:
            return None
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None
        best_idx = int(boxes.conf.argmax())
        xyxy = boxes.xyxy[best_idx].cpu().numpy()
        conf = float(boxes.conf[best_idx].cpu().numpy())
        return (tuple(map(float, xyxy)), conf)

    def _run_stage2(self, roi_crop: np.ndarray) -> List[Tuple[Tuple[float, float, float, float], float]]:
        """Run digit detection on ROI crop. Returns [(xyxy, conf), ...] in ROI pixel coords."""
        self._ensure_models()
        results = self._model_stage2.predict(
            roi_crop,
            conf=self.conf_digit,
            verbose=False,
            device=self.device,
        )
        if not results or len(results) == 0:
            return []
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []
        out = []
        for i in range(len(boxes)):
            xyxy = tuple(map(float, boxes.xyxy[i].cpu().numpy()))
            conf = float(boxes.conf[i].cpu().numpy())
            out.append((xyxy, conf))
        return out

    def _run_stage3(self, digit_crops: List[np.ndarray]) -> List[Tuple[int, float]]:
        """Run digit classification on each crop. Returns [(class_int, conf), ...]."""
        if not digit_crops:
            return []
        self._ensure_models()
        results = self._model_stage3.predict(
            digit_crops,
            verbose=False,
            device=self.device,
        )
        out = []
        for r in results:
            top1 = int(r.probs.top1)
            conf = float(r.probs.data[top1].cpu().numpy())
            out.append((top1, conf))
        return out

    def _crop_roi(
        self,
        image: np.ndarray,
        dial_box: Tuple[float, float, float, float],
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Crop image to dial ROI with padding. Returns (crop, (x1,y1,x2,y2) in img coords)."""
        h, w = image.shape[:2]
        x1, y1, x2, y2 = dial_box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        bw = x2 - x1
        bh = y2 - y1
        pad_w = bw * self.ROI_PADDING
        pad_h = bh * self.ROI_PADDING
        x1_p = max(0, int(x1 - pad_w))
        y1_p = max(0, int(y1 - pad_h))
        x2_p = min(w, int(x2 + pad_w))
        y2_p = min(h, int(y2 + pad_h))
        crop = image[y1_p:y2_p, x1_p:x2_p].copy()
        return crop, (x1_p, y1_p, x2_p, y2_p)

    def _crop_digit(
        self,
        roi_crop: np.ndarray,
        digit_box: Tuple[float, float, float, float],
    ) -> np.ndarray:
        """Crop a single digit from ROI and resize to 80x80."""
        rh, rw = roi_crop.shape[:2]
        x1, y1, x2, y2 = digit_box
        x1_i = max(0, int(x1))
        y1_i = max(0, int(y1))
        x2_i = min(rw, int(x2))
        y2_i = min(rh, int(y2))
        if x2_i <= x1_i or y2_i <= y1_i:
            return np.zeros((self.STAGE3_IMGSZ, self.STAGE3_IMGSZ, 3), dtype=np.uint8)
        digit = roi_crop[y1_i:y2_i, x1_i:x2_i]
        if not CV2_AVAILABLE:
            raise RuntimeError("OpenCV required for image resize")
        return cv2.resize(digit, (self.STAGE3_IMGSZ, self.STAGE3_IMGSZ))

    def _roi_to_img_coords(
        self,
        digit_box_roi: Tuple[float, float, float, float],
        roi_origin: Tuple[int, int, int, int],
    ) -> Tuple[float, float, float, float]:
        """Convert digit box from ROI pixel coords to original image pixel coords."""
        x1, y1, x2, y2 = digit_box_roi
        ox1, oy1, _, _ = roi_origin
        return (x1 + ox1, y1 + oy1, x2 + ox1, y2 + oy1)

    def _assemble_reading(
        self,
        digit_boxes_in_roi: List[Tuple[float, float, float, float]],
        digit_classes: List[int],
        digit_confidences: List[float],
    ) -> Tuple[List[Tuple[float, float, float, float]], List[int], List[float]]:
        """Sort by x_center (left-to-right) and return ordered lists."""
        if not digit_boxes_in_roi:
            return [], [], []
        centers = [(b[0] + b[2]) / 2 for b in digit_boxes_in_roi]
        order = sorted(range(len(centers)), key=lambda i: centers[i])
        boxes = [digit_boxes_in_roi[i] for i in order]
        classes = [digit_classes[i] for i in order]
        confs = [digit_confidences[i] for i in order]
        return boxes, classes, confs

    def predict(self, image: Union[str, Path, np.ndarray]) -> Optional[ReadingResult]:
        """
        Run full pipeline on a single image.
        Returns ReadingResult or None if no dial detected.
        """
        img = self._load_image(image)
        h, w = img.shape[:2]

        dial_out = self._run_stage1(img)
        if dial_out is None:
            return None

        dial_box, _ = dial_out
        roi_crop, roi_origin = self._crop_roi(img, dial_box)

        dial_conf = dial_out[1]

        digit_boxes_with_conf = self._run_stage2(roi_crop)
        if not digit_boxes_with_conf:
            return ReadingResult(
                reading="",
                confidence=0.0,
                dial_confidence=dial_conf,
                dial_box=dial_box,
                digit_boxes_in_roi=[],
                digit_boxes_in_img=[],
                digit_classes=[],
                digit_confidences=[],
                image_shape=(h, w),
            )

        digit_boxes_in_roi = [b[0] for b in digit_boxes_with_conf]
        digit_crops = [self._crop_digit(roi_crop, b) for b in digit_boxes_in_roi]
        digit_cls_results = self._run_stage3(digit_crops)

        digit_classes = [r[0] for r in digit_cls_results]
        digit_confidences = [r[1] for r in digit_cls_results]

        digit_boxes_in_roi, digit_classes, digit_confidences = self._assemble_reading(
            digit_boxes_in_roi, digit_classes, digit_confidences
        )

        digit_boxes_in_img = [
            self._roi_to_img_coords(b, roi_origin) for b in digit_boxes_in_roi
        ]

        reading = "".join(str(c) for c in digit_classes)
        mean_conf = sum(digit_confidences) / len(digit_confidences) if digit_confidences else 0.0

        return ReadingResult(
            reading=reading,
            confidence=mean_conf,
            dial_confidence=dial_conf,
            dial_box=dial_box,
            digit_boxes_in_roi=digit_boxes_in_roi,
            digit_boxes_in_img=digit_boxes_in_img,
            digit_classes=digit_classes,
            digit_confidences=digit_confidences,
            image_shape=(h, w),
        )

    def predict_batch(
        self,
        images: List[Union[str, Path, np.ndarray]],
    ) -> List[Optional[ReadingResult]]:
        """Run pipeline on multiple images."""
        return [self.predict(img) for img in images]
