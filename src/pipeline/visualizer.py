"""
Visualization utilities for pipeline inference results.
"""

from typing import Optional

import numpy as np

from .reader import ReadingResult

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


def draw_pipeline_result(
    image: np.ndarray,
    result: ReadingResult,
    show_dial_box: bool = True,
    show_digit_boxes: bool = True,
    show_reading_text: bool = True,
) -> np.ndarray:
    """
    Draw pipeline result annotations on the image.

    Args:
        image: BGR image (will be copied, not modified in place)
        result: ReadingResult from PipelineReader.predict
        show_dial_box: Draw green box around dial ROI
        show_digit_boxes: Draw cyan boxes around each digit with class label
        show_reading_text: Draw assembled reading string

    Returns:
        Annotated image (copy)
    """
    if not CV2_AVAILABLE:
        raise RuntimeError("OpenCV required for visualization. pip install opencv-python")

    out = image.copy()

    if show_dial_box and result.dial_box is not None:
        x1, y1, x2, y2 = result.dial_box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if show_reading_text and result.reading:
            label = f"Reading: {result.reading}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(out, (x1, y1 - th - 10), (x1 + tw + 4, y1), (0, 255, 0), -1)
            cv2.putText(
                out, label, (x1 + 2, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2,
            )

    if show_digit_boxes and result.digit_boxes_in_img:
        for i, box in enumerate(result.digit_boxes_in_img):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cls_val = result.digit_classes[i] if i < len(result.digit_classes) else 0
            conf_val = result.digit_confidences[i] if i < len(result.digit_confidences) else 0.0
            label = f"{cls_val} ({conf_val:.2f})"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), (255, 255, 0), -1)
            cv2.putText(
                out, label, (x1 + 2, y1 - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
            )

    return out
