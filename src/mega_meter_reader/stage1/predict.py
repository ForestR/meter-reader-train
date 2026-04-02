"""Stage 1 inference: segmentation + DIAL ROI."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np

from mega_meter_reader.stage1.postprocess import postprocess_stage1
from mega_meter_reader.stage1.types import Stage1Output


def run_stage1(
    model,
    image_bgr: np.ndarray,
    *,
    class_names: Optional[List[str]] = None,
    conf: float = 0.25,
    iou: float = 0.7,
    imgsz: Optional[int] = None,
    device: Optional[str] = None,
) -> Tuple[Stage1Output, Any]:
    """
    Run YOLO-seg on ``image_bgr`` and post-process to DIAL_ROI.

    Parameters
    ----------
    model :
        ``ultralytics.YOLO`` instance (segmentation weights, e.g. ``*-seg.pt``).
    image_bgr :
        ``uint8`` BGR image, shape ``(H, W, 3)``.

    Returns
    -------
    Stage1Output
        DIAL ROI and validity flags.
    Any
        Raw Ultralytics ``Results`` object (first element is used for masks).
    """
    kwargs = {"conf": conf, "iou": iou, "verbose": False}
    if imgsz is not None:
        kwargs["imgsz"] = imgsz
    if device is not None:
        kwargs["device"] = device

    results = model.predict(image_bgr, **kwargs)
    out = postprocess_stage1(image_bgr, results, class_names=class_names)
    out.results = results[0] if results else None
    return out, results
