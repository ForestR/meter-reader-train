"""
Manual labeler: Interactive OpenCV GUI for drawing/correcting YOLO bounding boxes.
Supports Stage 1 (dial bbox) and Stage 2 (digit bboxes in ROI crop).
Uses pipeline prediction as optional starting point.
"""

import os
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np

from .reader import PipelineReader, ReadingResult

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


def _check_display_available() -> None:
    """Raise RuntimeError if no display is available for OpenCV GUI (headless env)."""
    if os.name == "posix" and not os.environ.get("DISPLAY"):
        raise RuntimeError(
            "Manual labeling requires a display. No DISPLAY found (headless environment).\n"
            "Options: 1) Use X11 forwarding: ssh -X or ssh -Y\n"
            "         2) Use virtual framebuffer: xvfb-run python scripts/label_manual.py ..."
        )


def _xyxy_to_yolo_norm(
    x1: float, y1: float, x2: float, y2: float,
    img_w: int, img_h: int,
) -> Tuple[float, float, float, float]:
    """Convert pixel xyxy to normalized YOLO (cx, cy, w, h)."""
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return (cx, cy, w, h)


def _yolo_to_xyxy(
    cx: float, cy: float, w: float, h: float,
    img_w: int, img_h: int,
) -> Tuple[float, float, float, float]:
    """Convert normalized YOLO to pixel xyxy."""
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return (x1, y1, x2, y2)


def _read_yolo_label(path: Path, img_w: int, img_h: int) -> List[Tuple[float, float, float, float]]:
    """Read YOLO label file, return list of (x1,y1,x2,y2) in pixel coords."""
    boxes = []
    if not path.exists():
        return boxes
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                boxes.append(_yolo_to_xyxy(cx, cy, w, h, img_w, img_h))
            except (ValueError, IndexError):
                continue
    return boxes


class _LabelerState:
    """Mutable state for the OpenCV GUI callback."""

    def __init__(
        self,
        image: np.ndarray,
        boxes: List[Tuple[float, float, float, float]],
        hint_boxes: List[Tuple[float, float, float, float]],
        stage: Literal["stage1", "stage2"],
        show_hint: bool,
    ):
        self.image = image
        self.boxes = list(boxes)
        self.hint_boxes = list(hint_boxes)
        self.stage = stage
        self.show_hint = show_hint
        self.drawing = False
        self.start_pt: Optional[Tuple[int, int]] = None
        self.current_pt: Optional[Tuple[int, int]] = None
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.win_w = 1280
        self.win_h = 720
        self.display: Optional[np.ndarray] = None


def _draw_dashed_rect(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, color: Tuple[int, int, int], thickness: int = 2):
    """Draw a dashed rectangle by drawing alternating segments."""
    dash_len = 8
    for x in range(x1, x2, dash_len * 2):
        x_end = min(x + dash_len, x2)
        cv2.line(img, (x, y1), (x_end, y1), color, thickness)
    for x in range(x1, x2, dash_len * 2):
        x_end = min(x + dash_len, x2)
        cv2.line(img, (x, y2), (x_end, y2), color, thickness)
    for y in range(y1, y2, dash_len * 2):
        y_end = min(y + dash_len, y2)
        cv2.line(img, (x1, y), (x1, y_end), color, thickness)
    for y in range(y1, y2, dash_len * 2):
        y_end = min(y + dash_len, y2)
        cv2.line(img, (x2, y), (x2, y_end), color, thickness)


def _make_display(state: _LabelerState) -> np.ndarray:
    """Build the display image with boxes and status bar."""
    h, w = state.image.shape[:2]
    scale = min(state.win_w / w, (state.win_h - 40) / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    state.scale = scale
    state.offset_x = (state.win_w - new_w) // 2
    state.offset_y = 0

    canvas = np.zeros((state.win_h, state.win_w, 3), dtype=np.uint8)
    canvas[:] = (50, 50, 50)
    resized = cv2.resize(state.image, (new_w, new_h))
    canvas[state.offset_y : state.offset_y + new_h, state.offset_x : state.offset_x + new_w] = resized

    def scale_box(box: Tuple[float, float, float, float]) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = box
        return (
            state.offset_x + int(x1 * scale),
            state.offset_y + int(y1 * scale),
            state.offset_x + int(x2 * scale),
            state.offset_y + int(y2 * scale),
        )

    if state.show_hint and state.hint_boxes:
        for box in state.hint_boxes:
            x1, y1, x2, y2 = scale_box(box)
            _draw_dashed_rect(canvas, x1, y1, x2, y2, (0, 255, 255), 2)

    for i, box in enumerate(state.boxes):
        x1, y1, x2, y2 = scale_box(box)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if state.stage == "stage2":
            cv2.putText(canvas, str(i + 1), (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    if state.drawing and state.start_pt and state.current_pt:
        sx, sy = state.start_pt
        cx, cy = state.current_pt
        dx1 = state.offset_x + int(sx * scale)
        dy1 = state.offset_y + int(sy * scale)
        dx2 = state.offset_x + int(cx * scale)
        dy2 = state.offset_y + int(cy * scale)
        cv2.rectangle(canvas, (dx1, dy1), (dx2, dy2), (0, 255, 0), 2)

    status = "Enter:save r:reset d:del h:hint s:skip q:quit"
    if state.stage == "stage2":
        status += " | Draw digit boxes L->R"
    else:
        status += " | Draw dial box"
    cv2.rectangle(canvas, (0, state.win_h - 40), (state.win_w, state.win_h), (40, 40, 40), -1)
    cv2.putText(canvas, status, (10, state.win_h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    state.display = canvas
    return canvas


def _mouse_callback(event, x, y, flags, userdata):
    state: _LabelerState = userdata
    img_h, img_w = state.image.shape[:2]
    ix = int((x - state.offset_x) / state.scale) if state.scale else 0
    iy = int((y - state.offset_y) / state.scale) if state.scale else 0
    ix = max(0, min(img_w, ix))
    iy = max(0, min(img_h, iy))

    if event == cv2.EVENT_LBUTTONDOWN:
        state.drawing = True
        state.start_pt = (ix, iy)
        state.current_pt = (ix, iy)
    elif event == cv2.EVENT_MOUSEMOVE:
        if state.drawing:
            state.current_pt = (ix, iy)
    elif event == cv2.EVENT_LBUTTONUP:
        if state.drawing and state.start_pt:
            x1 = min(state.start_pt[0], ix)
            y1 = min(state.start_pt[1], iy)
            x2 = max(state.start_pt[0], ix)
            y2 = max(state.start_pt[1], iy)
            if x2 - x1 > 4 and y2 - y1 > 4:
                if state.stage == "stage1":
                    state.boxes = [(float(x1), float(y1), float(x2), float(y2))]
                else:
                    state.boxes.append((float(x1), float(y1), float(x2), float(y2)))
            state.drawing = False
            state.start_pt = None
            state.current_pt = None


class ManualLabeler:
    """
    Interactive OpenCV GUI for manual labeling.
    Supports Stage 1 (dial bbox) and Stage 2 (digit bboxes in ROI crop).
    """

    def __init__(
        self,
        stage: Literal["stage1", "stage2"],
        reader: Optional[PipelineReader] = None,
        window_size: Tuple[int, int] = (1280, 720),
    ):
        self.stage = stage
        self.reader = reader
        self.window_size = window_size
        if not CV2_AVAILABLE:
            raise RuntimeError("OpenCV required for manual labeling. pip install opencv-python")
        _check_display_available()

    def label_image(
        self,
        img_path: Path,
        existing_label: Optional[Path] = None,
        hint_result: Optional[ReadingResult] = None,
    ) -> Optional[List[Tuple[float, float, float, float]]]:
        """
        Interactively label one image. Returns list of YOLO (cx,cy,w,h) normalized boxes, or None if skipped.
        """
        img = cv2.imread(str(img_path))
        if img is None:
            return None
        h, w = img.shape[:2]

        hint_boxes: List[Tuple[float, float, float, float]] = []
        if hint_result:
            if self.stage == "stage1" and hint_result.dial_box:
                hint_boxes = [hint_result.dial_box]
            elif self.stage == "stage2" and hint_result.digit_boxes_in_roi:
                hint_boxes = list(hint_result.digit_boxes_in_roi)

        existing_boxes: List[Tuple[float, float, float, float]] = []
        if existing_label and existing_label.exists():
            existing_boxes = _read_yolo_label(existing_label, w, h)

        boxes = existing_boxes if existing_boxes else (list(hint_boxes) if hint_boxes else [])
        state = _LabelerState(img, boxes, hint_boxes, self.stage, show_hint=bool(hint_boxes))
        state.win_w, state.win_h = self.window_size

        win_name = "Manual Labeler"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, state.win_w, state.win_h)
        cv2.setMouseCallback(win_name, _mouse_callback, state)

        result_boxes: Optional[List[Tuple[float, float, float, float]]] = None

        while True:
            disp = _make_display(state)
            cv2.imshow(win_name, disp)
            k = cv2.waitKey(30) & 0xFF
            if k == ord("q"):
                result_boxes = None
                break
            if k == ord("s"):
                result_boxes = None
                break
            if k == ord("r"):
                state.boxes = []
            if k == ord("d") and state.stage == "stage2" and state.boxes:
                state.boxes.pop()
            if k == ord("h"):
                state.show_hint = not state.show_hint
            if k in (13, 32):
                if state.boxes:
                    if self.stage == "stage1":
                        if len(state.boxes) >= 1:
                            b = state.boxes[0]
                            result_boxes = [_xyxy_to_yolo_norm(b[0], b[1], b[2], b[3], w, h)]
                            break
                    else:
                        result_boxes = [
                            _xyxy_to_yolo_norm(b[0], b[1], b[2], b[3], w, h)
                            for b in state.boxes
                        ]
                        break
                else:
                    result_boxes = None
                    break

        cv2.destroyWindow(win_name)
        return result_boxes

    def label_directory(
        self,
        src_images: Path,
        dst: Path,
        resume: bool = True,
        use_pipeline_hint: bool = True,
        review: bool = False,
        stage1_labels_dir: Optional[Path] = None,
    ) -> Dict[str, int]:
        """
        Label all images in directory. Returns {"labeled": N, "skipped": N, "quit": 0|1}.
        For stage2, stage1_labels_dir must point to directory containing labels from stage1 (to crop ROI).
        """
        images_dir = src_images / "images" if (src_images / "images").exists() else src_images
        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")

        ext = {".jpg", ".jpeg", ".png", ".bmp"}
        image_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in ext)

        dst_images = dst / "images"
        dst_labels = dst / "labels"
        dst_images.mkdir(parents=True, exist_ok=True)
        dst_labels.mkdir(parents=True, exist_ok=True)

        if self.stage == "stage2" and stage1_labels_dir is None:
            stage1_labels_dir = dst.parent / "labels" if (dst.parent / "labels").exists() else None

        stats = {"labeled": 0, "skipped": 0, "quit": 0}
        total = len(image_paths)

        for idx, img_path in enumerate(image_paths):
            stem = img_path.stem
            label_path = dst_labels / f"{stem}.txt"
            if resume and not review and label_path.exists():
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                stats["skipped"] += 1
                continue

            h, w = img.shape[:2]
            work_image = img
            work_w, work_h = w, h

            hint_result: Optional[ReadingResult] = None
            if use_pipeline_hint and self.reader:
                try:
                    hint_result = self.reader.predict(str(img_path))
                except Exception:
                    pass

            if self.stage == "stage2":
                dial_box: Optional[Tuple[float, float, float, float]] = None
                if hint_result and hint_result.dial_box:
                    dial_box = hint_result.dial_box
                elif stage1_labels_dir:
                    s1_label = stage1_labels_dir / f"{stem}.txt"
                    if s1_label.exists():
                        boxes = _read_yolo_label(s1_label, w, h)
                        if boxes:
                            dial_box = boxes[0]
                if dial_box is None:
                    dial_phase_state = _LabelerState(img, [], [], "stage1", show_hint=bool(hint_result and hint_result.dial_box))
                    dial_phase_state.win_w, dial_phase_state.win_h = self.window_size
                    if hint_result and hint_result.dial_box:
                        dial_phase_state.boxes = [hint_result.dial_box]
                        dial_phase_state.hint_boxes = [hint_result.dial_box]
                    win_name = "Manual Labeler - Draw dial"
                    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(win_name, dial_phase_state.win_w, dial_phase_state.win_h)
                    cv2.setMouseCallback(win_name, _mouse_callback, dial_phase_state)
                    while True:
                        disp = _make_display(dial_phase_state)
                        cv2.imshow(win_name, disp)
                        k = cv2.waitKey(30) & 0xFF
                        if k == ord("q"):
                            stats["quit"] = 1
                            cv2.destroyAllWindows()
                            return stats
                        if k == ord("s"):
                            stats["skipped"] += 1
                            cv2.destroyWindow(win_name)
                            break
                        if k in (13, 32) and dial_phase_state.boxes:
                            dial_box = dial_phase_state.boxes[0]
                            cv2.destroyWindow(win_name)
                            break
                    if dial_box is None:
                        continue
                roi_crop, roi_origin = self.reader._crop_roi(img, dial_box) if self.reader else self._crop_roi_fallback(img, dial_box)
                work_image = roi_crop
                work_h, work_w = roi_crop.shape[:2]
                if hint_result and hint_result.digit_boxes_in_roi:
                    hint_boxes = hint_result.digit_boxes_in_roi
                else:
                    hint_boxes = []
            else:
                hint_boxes = [hint_result.dial_box] if (hint_result and hint_result.dial_box) else []

            existing: Optional[Path] = label_path if review else None
            state = _LabelerState(work_image, [], hint_boxes, self.stage, show_hint=use_pipeline_hint and bool(hint_boxes))
            state.win_w, state.win_h = self.window_size
            if existing and existing.exists():
                state.boxes = _read_yolo_label(existing, work_w, work_h)
            elif hint_boxes:
                state.boxes = list(hint_boxes)

            win_name = f"Manual Labeler [{idx + 1}/{total}] {img_path.name}"
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win_name, state.win_w, state.win_h)
            cv2.setMouseCallback(win_name, _mouse_callback, state)

            result_boxes = None
            while True:
                disp = _make_display(state)
                cv2.imshow(win_name, disp)
                k = cv2.waitKey(30) & 0xFF
                if k == ord("q"):
                    stats["quit"] = 1
                    cv2.destroyAllWindows()
                    return stats
                if k == ord("s"):
                    stats["skipped"] += 1
                    break
                if k == ord("r"):
                    state.boxes = []
                if k == ord("d") and self.stage == "stage2" and state.boxes:
                    state.boxes.pop()
                if k == ord("h"):
                    state.show_hint = not state.show_hint
                if k in (13, 32):
                    if state.boxes:
                        result_boxes = [
                            _xyxy_to_yolo_norm(b[0], b[1], b[2], b[3], work_w, work_h)
                            for b in state.boxes
                        ]
                    break

            cv2.destroyWindow(win_name)

            if result_boxes:
                with open(label_path, "w") as f:
                    for cx, cy, bw, bh in result_boxes:
                        f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                out_img = dst_images / f"{stem}{img_path.suffix}"
                if self.stage == "stage1":
                    import shutil
                    shutil.copy2(img_path, out_img)
                else:
                    cv2.imwrite(str(out_img), work_image)
                stats["labeled"] += 1

        return stats

    def _crop_roi_fallback(
        self,
        image: np.ndarray,
        dial_box: Tuple[float, float, float, float],
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Fallback ROI crop when reader is None (uses same logic as PipelineReader)."""
        h, w = image.shape[:2]
        x1, y1, x2, y2 = dial_box
        padding = 0.05
        bw = x2 - x1
        bh = y2 - y1
        pad_w = bw * padding
        pad_h = bh * padding
        x1_p = max(0, int(x1 - pad_w))
        y1_p = max(0, int(y1 - pad_h))
        x2_p = min(w, int(x2 + pad_w))
        y2_p = min(h, int(y2 + pad_h))
        crop = image[y1_p:y2_p, x1_p:x2_p].copy()
        return crop, (x1_p, y1_p, x2_p, y2_p)
