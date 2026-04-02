"""Tight canvas: output ROI should not be much larger than warped union silhouette."""

from __future__ import annotations

import numpy as np
import cv2
import pytest

from mega_meter_reader.stage1.postprocess import build_dial_roi, DEFAULT_PAD_COLOR


def _warped_union_bbox_hw(
    dial: np.ndarray,
    decimal: np.ndarray,
    affine_same_as_warp: np.ndarray,
    out_hw: tuple[int, int],
) -> tuple[int, int]:
    """Tight (width, height) of union mask in ROI space (same matrix as ``warpAffine`` on crop)."""
    u = (np.maximum(dial, decimal) > 0.5).astype(np.uint8) * 255
    oh, ow = out_hw
    warped = cv2.warpAffine(
        u, affine_same_as_warp.astype(np.float32), (ow, oh), flags=cv2.INTER_NEAREST
    )
    ys, xs = np.where(warped > 127)
    assert len(xs) > 0, "empty warped union"
    bw = int(xs.max() - xs.min() + 1)
    bh = int(ys.max() - ys.min() + 1)
    return bw, bh


def _synthetic_scene(
    h: int = 640,
    w: int = 640,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gray BGR image + dial / decimal float masks (horizontal bar + right decimal patch)."""
    img = np.full((h, w, 3), 200, dtype=np.uint8)
    dial = np.zeros((h, w), dtype=np.float32)
    dec = np.zeros((h, w), dtype=np.float32)
    # Small union inside a large canvas (crop will be loose before tight output).
    dial[220:260, 120:520] = 1.0
    dec[228:252, 400:480] = 1.0
    return img, dial, dec


@pytest.mark.parametrize("margin_ratio", [0.05, 0.08])
def test_dial_roi_canvas_at_most_150pct_union_bbox(margin_ratio: float) -> None:
    """Canvas W/H should not exceed 150% of warped union tight bbox per axis."""
    img, dial, dec = _synthetic_scene()
    out = build_dial_roi(
        img,
        dial,
        dec,
        margin_ratio=margin_ratio,
        pad_color=DEFAULT_PAD_COLOR,
    )
    assert not out.is_invalid, out.warnings
    oh, ow = out.dial_roi.shape[:2]
    bw, bh = _warped_union_bbox_hw(dial, dec, out.affine_matrix, (oh, ow))
    eps = 4.0  # rounding / interpolation slack
    assert ow <= 1.5 * bw + eps, f"ow={ow} bw={bw}"
    assert oh <= 1.5 * bh + eps, f"oh={oh} bh={bh}"


def test_tight_canvas_smaller_than_full_rotated_crop() -> None:
    """Union much smaller than crop: output should be tighter than old full-crop canvas."""
    img, dial, dec = _synthetic_scene()
    out = build_dial_roi(img, dial, dec, margin_ratio=0.05)
    assert not out.is_invalid
    oh, ow = out.dial_roi.shape[:2]
    # Loose crop is ~union bbox + margin; union is ~400x40 in 640x640 — crop still large.
    # Tight output should be far below full-image area.
    assert ow * oh < 0.5 * img.shape[0] * img.shape[1]
