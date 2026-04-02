"""
DIAL ROI: validate dial + decimal_section masks and produce a rotated, padded crop.

See docs/architecture.md (Stage 1 post-processing).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from mega_meter_reader.stage1.types import Stage1Output

# BGR padding (green background)
DEFAULT_PAD_COLOR = (0, 255, 0)


def _centroid_from_mask(mask: np.ndarray) -> Tuple[float, float]:
    m = (mask > 0.5).astype(np.uint8)
    M = cv2.moments(m)
    if M["m00"] < 1e-6:
        return float("nan"), float("nan")
    return float(M["m10"] / M["m00"]), float(M["m01"] / M["m00"])


def _tight_rotated_union_aabb(
    union_u8: np.ndarray,
    M0: np.ndarray,
    cw: int,
    ch: int,
) -> Tuple[float, float, float, float]:
    """
    Axis-aligned bounds in M0-rotated space for union foreground (crop coordinates).

    Uses contour vertices (cheap); falls back to rotated full crop corners if empty.
    Returns qx_min, qx_max, qy_min, qy_max.
    """
    contours, _ = cv2.findContours(
        union_u8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        corners = np.array(
            [[0, 0], [cw, 0], [cw, ch], [0, ch]], dtype=np.float32
        )
        rot_c = cv2.transform(corners.reshape(1, -1, 2), M0).reshape(-1, 2)
        return (
            float(rot_c[:, 0].min()),
            float(rot_c[:, 0].max()),
            float(rot_c[:, 1].min()),
            float(rot_c[:, 1].max()),
        )
    cnt = max(contours, key=cv2.contourArea)
    if cnt.size < 2:
        corners = np.array(
            [[0, 0], [cw, 0], [cw, ch], [0, ch]], dtype=np.float32
        )
        rot_c = cv2.transform(corners.reshape(1, -1, 2), M0).reshape(-1, 2)
        return (
            float(rot_c[:, 0].min()),
            float(rot_c[:, 0].max()),
            float(rot_c[:, 1].min()),
            float(rot_c[:, 1].max()),
        )
    pts = cnt.reshape(-1, 2).astype(np.float32)
    warped = cv2.transform(pts.reshape(-1, 1, 2), M0).reshape(-1, 2)
    return (
        float(warped[:, 0].min()),
        float(warped[:, 0].max()),
        float(warped[:, 1].min()),
        float(warped[:, 1].max()),
    )


def masks_from_segmentation_result(
    results,
    orig_hw: Tuple[int, int],
    class_names: Optional[List[str]] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[str]]:
    """
    Build dense dial / decimal masks (H, W) float32 in [0,1] from first Ultralytics Results.

    Picks highest-confidence detection per class id (0=dial, 1=decimal_section).
    """
    if class_names is not None:
        # Reserved for custom ``names`` order vs. default [dial, decimal_section].
        pass
    warnings: List[str] = []
    H, W = orig_hw
    r = results[0]
    if r.masks is None or r.boxes is None or len(r.boxes) == 0:
        warnings.append("No masks or boxes in model output.")
        return None, None, warnings

    cls_ids = r.boxes.cls.cpu().numpy().astype(int)
    confs = r.boxes.conf.cpu().numpy()
    md = r.masks.data.cpu().numpy()  # n, mh, mw

    dial_mask = np.zeros((H, W), dtype=np.float32)
    dec_mask = np.zeros((H, W), dtype=np.float32)

    best_dial = (-1.0, None)
    best_dec = (-1.0, None)
    for i in range(len(cls_ids)):
        c = int(cls_ids[i])
        if c == 0:
            if confs[i] > best_dial[0]:
                best_dial = (float(confs[i]), i)
        elif c == 1:
            if confs[i] > best_dec[0]:
                best_dec = (float(confs[i]), i)

    def upscale(mi: int) -> np.ndarray:
        m = md[mi]
        return cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)

    if best_dial[1] is None:
        warnings.append("No dial (class 0) instance found.")
    else:
        dial_mask = upscale(best_dial[1])

    if best_dec[1] is None:
        warnings.append("No decimal_section (class 1) instance found.")
    else:
        dec_mask = upscale(best_dec[1])

    return dial_mask, dec_mask, warnings


def validate_masks(
    dial: np.ndarray,
    decimal: np.ndarray,
) -> Tuple[bool, List[str]]:
    """Exactly one connected component each with non-trivial area; non-empty intersection."""
    warnings: List[str] = []
    d = (dial > 0.5).astype(np.uint8)
    e = (decimal > 0.5).astype(np.uint8)
    if d.sum() == 0 or e.sum() == 0:
        warnings.append("Empty dial or decimal mask after thresholding.")
        return False, warnings
    n_d, _ = cv2.connectedComponents(d)
    n_e, _ = cv2.connectedComponents(e)
    if n_d - 1 != 1:
        warnings.append(f"Expected 1 dial component, found {n_d - 1}.")
    if n_e - 1 != 1:
        warnings.append(f"Expected 1 decimal_section component, found {n_e - 1}.")
    inter = np.logical_and(d > 0, e > 0)
    if not inter.any():
        warnings.append("dial and decimal_section masks do not intersect.")
        return False, warnings
    if n_d - 1 != 1 or n_e - 1 != 1:
        return False, warnings
    return True, warnings


def pca_long_axis_alignment(
    mask: np.ndarray,
    *,
    ratio_threshold: float = 2.0,
    min_points: int = 10,
) -> Tuple[float, Tuple[float, float], float]:
    """
    Align the union mask elongation with the horizontal axis.

    Uses ``cv2.PCACompute2`` eigenvalues for sqrt(lambda1/lambda2) (anisotropy) and the
    central second-moment orientation
    ``0.5 * atan2(2 sigma_xy, sigma_xx - sigma_yy)`` in ``(-90°, 90°]``, which matches
    OpenCV rotation and avoids mixing up the major PC with the geometric long axis when
    ``sigma_xx ~= sigma_yy``.

    :param mask: Dense mask (H, W), float in [0,1] or uint8; foreground is ``> 0.5``.
    :param ratio_threshold: Minimum sqrt(lambda_major / lambda_minor); below this, degenerate.
    :param min_points: Minimum foreground pixels to run PCA.
    :return: (angle_deg, center_xy, aspect_ratio) for ``cv2.getRotationMatrix2D`` (CCW positive).
    :raises ValueError: Too few points or aspect ratio below threshold.
    """
    m = (mask > 0.5).astype(np.uint8)
    coords = np.column_stack(np.where(m > 0))
    if len(coords) < min_points:
        raise ValueError(f"Too few mask points ({len(coords)} < {min_points}).")
    pts = coords[:, [1, 0]].astype(np.float32)
    mean, _, eigenvalues = cv2.PCACompute2(pts, mean=None)
    major_val = float(eigenvalues[0, 0])
    minor_val = float(eigenvalues[1, 0])
    if minor_val < 1e-7:
        aspect_ratio = float("inf")
    else:
        aspect_ratio = float(np.sqrt(major_val / minor_val))
    if aspect_ratio < ratio_threshold:
        raise ValueError(
            f"Degenerate object: PCA aspect ratio {aspect_ratio:.2f} < {ratio_threshold}."
        )
    mu = mean[0].astype(np.float64)
    centered = pts.astype(np.float64) - mu
    c = np.cov(centered[:, 0], centered[:, 1])
    cxx, cyy, cxy = float(c[0, 0]), float(c[1, 1]), float(c[0, 1])
    # Orientation in (-90°, 90°] matching OpenCV; avoids first-PC atan2 vs long-axis mismatch.
    angle_deg = float(
        0.5 * np.degrees(np.arctan2(2.0 * cxy, cxx - cyy))
    )
    center = (float(mean[0, 0]), float(mean[0, 1]))
    return angle_deg, center, aspect_ratio


def build_dial_roi(
    image_bgr: np.ndarray,
    dial_mask: np.ndarray,
    decimal_mask: np.ndarray,
    *,
    margin_ratio: float = 0.05,
    pad_color: Tuple[int, int, int] = DEFAULT_PAD_COLOR,
    pca_ratio_threshold: float = 2.0,
    min_mask_points: int = 10,
) -> Stage1Output:
    """
    Union crop using axis-aligned bbox of (dial|decimal), PCA-align the union's long axis
    to horizontal, size the output canvas from the rotated union silhouette (tight AABB
    + per-axis padding from ``margin_ratio``), then optionally flip 180° so decimal lies
    to the right of dial.
    """
    warnings: List[str] = []
    H, W = image_bgr.shape[:2]
    U = np.maximum(dial_mask, decimal_mask)
    u8 = (U > 0.5).astype(np.uint8) * 255
    ys, xs = np.where(u8 > 0)
    if len(xs) == 0:
        out = Stage1Output(
            dial_roi=np.copy(image_bgr),
            affine_matrix=np.eye(2, 3, dtype=np.float32),
            is_invalid=True,
            warnings=["Empty union mask."],
        )
        return out

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    m = int(max(bw, bh) * margin_ratio)
    x0 = max(0, x0 - m)
    y0 = max(0, y0 - m)
    x1 = min(W - 1, x1 + m)
    y1 = min(H - 1, y1 + m)

    crop = image_bgr[y0 : y1 + 1, x0 : x1 + 1].copy()
    dm = dial_mask[y0 : y1 + 1, x0 : x1 + 1]
    em = decimal_mask[y0 : y1 + 1, x0 : x1 + 1]

    cd = _centroid_from_mask(dm)
    ce = _centroid_from_mask(em)
    if np.isnan(cd[0]) or np.isnan(ce[0]):
        warnings.append("Invalid centroids.")
        return Stage1Output(
            dial_roi=crop,
            affine_matrix=np.array(
                [[1, 0, -float(x0)], [0, 1, -float(y0)]], dtype=np.float32
            ),
            is_invalid=True,
            warnings=warnings,
        )

    vx, vy = ce[0] - cd[0], ce[1] - cd[1]
    if vx * vx + vy * vy < 1e-6:
        warnings.append("Dial and decimal centroids coincide.")
        return Stage1Output(
            dial_roi=crop,
            affine_matrix=np.array(
                [[1, 0, -float(x0)], [0, 1, -float(y0)]], dtype=np.float32
            ),
            is_invalid=True,
            warnings=warnings,
        )

    U_crop = np.maximum(dm, em)
    try:
        angle_deg, center, _ = pca_long_axis_alignment(
            U_crop,
            ratio_threshold=pca_ratio_threshold,
            min_points=min_mask_points,
        )
    except ValueError as e:
        warnings.append(str(e))
        return Stage1Output(
            dial_roi=crop,
            affine_matrix=np.array(
                [[1, 0, -float(x0)], [0, 1, -float(y0)]], dtype=np.float32
            ),
            is_invalid=True,
            warnings=warnings,
        )

    ch, cw = crop.shape[:2]
    M0 = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    union_u8 = (U_crop > 0.5).astype(np.uint8) * 255
    qx_min, qx_max, qy_min, qy_max = _tight_rotated_union_aabb(
        union_u8, M0, cw, ch
    )
    content_w = max(1.0, qx_max - qx_min)
    content_h = max(1.0, qy_max - qy_min)
    pad_x = int(content_w * margin_ratio) + 2
    pad_y = int(content_h * margin_ratio) + 2
    out_w = int(np.ceil(content_w + 2 * pad_x))
    out_h = int(np.ceil(content_h + 2 * pad_y))
    M_rot = M0.copy()
    M_rot[0, 2] += float(pad_x) - qx_min
    M_rot[1, 2] += float(pad_y) - qy_min

    dial_roi = cv2.warpAffine(
        crop,
        M_rot,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=pad_color,
    )

    # Forward affine from original image coords (x,y) to dial_roi: p_crop = p - (x0,y0), then M_rot @ (p_crop,1)
    T_pre = np.array([[1, 0, -float(x0)], [0, 1, -float(y0)], [0, 0, 1]], dtype=np.float64)
    M3 = np.eye(3, dtype=np.float64)
    M3[:2, :] = M_rot.astype(np.float64)
    H_fwd = M3 @ T_pre
    affine_2x3 = H_fwd[:2, :].astype(np.float32)

    # Verify decimal is to the right of dial in output (optional flip 180)
    dm_r = cv2.warpAffine(
        (dm * 255).astype(np.uint8),
        M_rot,
        (out_w, out_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    em_r = cv2.warpAffine(
        (em * 255).astype(np.uint8),
        M_rot,
        (out_w, out_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    # Union mask outside silhouette -> pad color (not bbox-only background)
    u_roi = np.maximum(dm_r, em_r)
    outside = u_roi < 127
    dial_roi[outside] = pad_color

    cd2 = _centroid_from_mask(dm_r.astype(np.float32) / 255.0)
    ce2 = _centroid_from_mask(em_r.astype(np.float32) / 255.0)
    if not np.isnan(cd2[0]) and not np.isnan(ce2[0]) and ce2[0] < cd2[0]:
        M_flip = np.array(
            [[-1.0, 0.0, float(out_w - 1)], [0.0, -1.0, float(out_h - 1)]],
            dtype=np.float32,
        )
        dial_roi = cv2.warpAffine(
            dial_roi,
            M_flip,
            (out_w, out_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=pad_color,
        )
        H_flip = np.eye(3, dtype=np.float64)
        H_flip[0, 0] = -1
        H_flip[0, 2] = out_w - 1
        H_flip[1, 1] = -1
        H_flip[1, 2] = out_h - 1
        H_fwd = H_flip @ H_fwd
        affine_2x3 = H_fwd[:2, :].astype(np.float32)

    return Stage1Output(
        dial_roi=dial_roi,
        affine_matrix=affine_2x3,
        is_invalid=False,
        warnings=warnings,
    )


def postprocess_stage1(
    image_bgr: np.ndarray,
    results,
    *,
    class_names: Optional[List[str]] = None,
) -> Stage1Output:
    """
    Full validation + DIAL ROI from raw BGR image and Ultralytics Results.
    """
    h, w = image_bgr.shape[:2]
    dial, dec, w1 = masks_from_segmentation_result(results, (h, w), class_names)
    warnings = list(w1)
    if dial is None or dec is None:
        return Stage1Output(
            dial_roi=np.copy(image_bgr),
            affine_matrix=np.eye(2, 3, dtype=np.float32),
            is_invalid=True,
            warnings=warnings,
        )

    ok, w2 = validate_masks(dial, dec)
    warnings.extend(w2)
    if not ok:
        return Stage1Output(
            dial_roi=np.copy(image_bgr),
            affine_matrix=np.eye(2, 3, dtype=np.float32),
            is_invalid=True,
            warnings=warnings,
        )

    out = build_dial_roi(image_bgr, dial, dec)
    out.warnings = warnings + out.warnings
    return out
