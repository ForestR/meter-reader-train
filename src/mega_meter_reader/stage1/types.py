"""Stage 1 output types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np


@dataclass
class Stage1Output:
    """Result of Stage 1 inference + DIAL ROI post-processing."""

    dial_roi: np.ndarray  # BGR image (H, W, 3)
    affine_matrix: np.ndarray  # 2×3 float32, maps points in RAW image to DIAL_ROI: dst = M @ [x,y,1].T
    is_invalid: bool
    warnings: List[str] = field(default_factory=list)
    results: Optional[Any] = None  # optional Ultralytics Results for debugging
