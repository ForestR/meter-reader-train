"""
Angle Correction Utilities for Meter Reading
Handles rotation augmentation and alignment preprocessing.
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def rotate_image_with_bbox(image: np.ndarray, 
                           bbox: np.ndarray,
                           angle: float,
                           scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rotate image and adjust bounding box coordinates.
    
    Args:
        image: Input image (H, W, C)
        bbox: Bounding box in YOLO format [class, x_center, y_center, width, height]
        angle: Rotation angle in degrees (positive = counter-clockwise)
        scale: Scaling factor
        
    Returns:
        Tuple of (rotated_image, rotated_bbox)
    """
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    
    # Get rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, scale)
    
    # Calculate new image dimensions
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    # Adjust rotation matrix for new dimensions
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    
    # Rotate image
    rotated = cv2.warpAffine(image, M, (new_w, new_h), 
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(114, 114, 114))
    
    # Rotate bounding box
    class_id = bbox[0]
    x_center = bbox[1] * w
    y_center = bbox[2] * h
    box_w = bbox[3] * w
    box_h = bbox[4] * h
    
    # Get box corners
    corners = np.array([
        [x_center - box_w/2, y_center - box_h/2, 1],
        [x_center + box_w/2, y_center - box_h/2, 1],
        [x_center + box_w/2, y_center + box_h/2, 1],
        [x_center - box_w/2, y_center + box_h/2, 1]
    ]).T
    
    # Apply rotation
    rotated_corners = M @ corners
    rotated_corners = rotated_corners.T
    
    # Get new bounding box
    x_min = rotated_corners[:, 0].min()
    x_max = rotated_corners[:, 0].max()
    y_min = rotated_corners[:, 1].min()
    y_max = rotated_corners[:, 1].max()
    
    # Convert back to YOLO format (normalized)
    new_x_center = ((x_min + x_max) / 2) / new_w
    new_y_center = ((y_min + y_max) / 2) / new_h
    new_w_norm = (x_max - x_min) / new_w
    new_h_norm = (y_max - y_min) / new_h
    
    # Clip to [0, 1]
    new_x_center = np.clip(new_x_center, 0, 1)
    new_y_center = np.clip(new_y_center, 0, 1)
    new_w_norm = np.clip(new_w_norm, 0, 1)
    new_h_norm = np.clip(new_h_norm, 0, 1)
    
    rotated_bbox = np.array([class_id, new_x_center, new_y_center, new_w_norm, new_h_norm])
    
    return rotated, rotated_bbox


def estimate_rotation_angle(image: np.ndarray, 
                            method: str = 'hough') -> float:
    """
    Estimate rotation angle of a skewed meter display.
    Useful for inference-time alignment (future: pipeline Stage 1).
    
    Args:
        image: Input image (H, W, C)
        method: Detection method ('hough' or 'contours')
        
    Returns:
        Estimated rotation angle in degrees
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    if method == 'hough':
        # Hough Line Transform
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
        
        if lines is None or len(lines) == 0:
            return 0.0
        
        # Compute average angle
        angles = []
        for line in lines:
            rho, theta = line[0]
            angle = np.degrees(theta) - 90
            
            # Filter near-horizontal lines (±15° from horizontal)
            if abs(angle) < 15:
                angles.append(angle)
        
        if len(angles) == 0:
            return 0.0
        
        return np.median(angles)
    
    elif method == 'contours':
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return 0.0
        
        # Find largest contour (likely the meter display)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get minimum area rectangle
        rect = cv2.minAreaRect(largest_contour)
        angle = rect[2]
        
        # Adjust angle to [-15, 15] range
        if angle < -45:
            angle += 90
        elif angle > 45:
            angle -= 90
        
        return angle
    
    else:
        raise ValueError(f"Unknown method: {method}")


def apply_angle_correction(image: np.ndarray, 
                          angle: Optional[float] = None) -> np.ndarray:
    """
    Apply angle correction to straighten a skewed meter image.
    
    Args:
        image: Input image (H, W, C)
        angle: Rotation angle in degrees (if None, auto-detect)
        
    Returns:
        Corrected image
    """
    if angle is None:
        # Auto-detect rotation angle
        angle = estimate_rotation_angle(image)
    
    # Rotate image
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate new dimensions to avoid cropping
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    # Adjust rotation matrix
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    
    # Apply rotation
    corrected = cv2.warpAffine(image, M, (new_w, new_h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(114, 114, 114))
    
    return corrected


# Note: For Phase 1 (End-to-End training), angle correction is handled
# via YOLO's built-in rotation augmentation (degrees: 15.0 in config).
# 
# The functions in this module are for:
# 1. Custom preprocessing pipelines (future)
# 2. Inference-time alignment (Pipeline Stage 1, future)
# 3. Data augmentation validation/testing
#
# For training, use the configuration in:
# - configs/train_phase1_frozen.yaml
# - configs/train_phase2_unfrozen.yaml
