"""Synthetic tests for PCA long-axis alignment in stage1 postprocess."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from mega_meter_reader.stage1.postprocess import pca_long_axis_alignment


def _make_horizontal_bar_mask(h: int, w: int, bar_w: int, bar_h: int) -> np.ndarray:
    """Float mask [0,1] with a centered horizontal bar (major axis along +x)."""
    mask = np.zeros((h, w), dtype=np.float32)
    x0 = (w - bar_w) // 2
    y0 = (h - bar_h) // 2
    mask[y0 : y0 + bar_h, x0 : x0 + bar_w] = 1.0
    return mask


def _rotate_mask(
    mask: np.ndarray,
    angle_deg: float,
    center: tuple[float, float],
) -> np.ndarray:
    h, w = mask.shape[:2]
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(
        mask,
        M,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )


@pytest.mark.parametrize(
    "theta_deg",
    [0.0, 30.0, 45.0, 60.0, 90.0],
)
def test_pca_recovers_horizontal_after_align(theta_deg: float) -> None:
    """Rotated elongated rectangle: PCA + warp should leave major axis ~ horizontal."""
    h, w = 400, 400
    center = (w * 0.5, h * 0.5)
    base = _make_horizontal_bar_mask(h, w, 280, 22)
    rotated = _rotate_mask(base, theta_deg, center)

    angle_deg, ctr, aspect_ratio = pca_long_axis_alignment(
        rotated, ratio_threshold=2.0, min_points=10
    )
    assert aspect_ratio >= 2.0
    M = cv2.getRotationMatrix2D(ctr, angle_deg, 1.0)
    aligned = cv2.warpAffine(
        rotated,
        M,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )

    angle2, _, _ = pca_long_axis_alignment(
        aligned, ratio_threshold=2.0, min_points=10
    )
    # Second alignment should need ~0° rotation (±180° equivalent).
    d = abs(angle2) % 180.0
    d = min(d, 180.0 - d)
    tol = 3.0 if abs(theta_deg - 90.0) < 1e-6 else 2.0
    assert d < tol, f"theta_in={theta_deg} angle_after_align={angle2}"


def test_degenerate_square_raises() -> None:
    mask = np.zeros((200, 200), dtype=np.float32)
    mask[60:140, 60:140] = 1.0
    with pytest.raises(ValueError, match="aspect ratio"):
        pca_long_axis_alignment(mask, ratio_threshold=2.0, min_points=10)


def test_too_few_points_raises() -> None:
    mask = np.zeros((50, 50), dtype=np.float32)
    mask[24:26, 24:26] = 1.0
    with pytest.raises(ValueError, match="Too few mask points"):
        pca_long_axis_alignment(mask, ratio_threshold=1.1, min_points=100)
