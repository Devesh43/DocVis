import cv2
import numpy as np
from typing import Tuple

def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Orders 4 spatial coordinates into top-left, top-right, bottom-right, bottom-left order.
    
    Args:
        pts: Array of shape (4, 2) or (4, 1, 2).
        
    Returns:
        np.ndarray of shape (4, 2) with float32 type:
        [0]: Top-Left (TL)
        [1]: Top-Right (TR)
        [2]: Bottom-Right (BR)
        [3]: Bottom-Left (BL)
    """
    pts_reshaped = np.asarray(pts, dtype="float32").reshape((4, 2))
    rect = np.zeros((4, 2), dtype="float32")
    
    # Top-left has smallest sum (x + y), bottom-right has largest sum (x + y)
    s = pts_reshaped.sum(axis=1)
    rect[0] = pts_reshaped[np.argmin(s)]
    rect[2] = pts_reshaped[np.argmax(s)]
    
    # Top-right has smallest difference (y - x), bottom-left has largest difference (y - x)
    diff = np.diff(pts_reshaped, axis=1)
    rect[1] = pts_reshaped[np.argmin(diff)]
    rect[3] = pts_reshaped[np.argmax(diff)]
    
    return rect

def compute_target_dimensions(ordered_pts: np.ndarray) -> Tuple[int, int]:
    """
    Calculates the maximum width and height of the transformed document based on corner distances.
    
    Args:
        ordered_pts: np.ndarray of shape (4, 2) ordered [TL, TR, BR, BL]
        
    Returns:
        (max_width, max_height) in pixels.
    """
    (tl, tr, br, bl) = ordered_pts
    
    # Calculate widths (top edge and bottom edge)
    width_top = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    width_bottom = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    max_width = max(int(round(width_top)), int(round(width_bottom)))
    
    # Calculate heights (left edge and right edge)
    height_left = np.sqrt(((bl[0] - tl[0]) ** 2) + ((bl[1] - tl[1]) ** 2))
    height_right = np.sqrt(((br[0] - tr[0]) ** 2) + ((br[1] - tr[1]) ** 2))
    max_height = max(int(round(height_left)), int(round(height_right)))
    
    # Prevent degenerate dimensions
    max_width = max(max_width, 100)
    max_height = max(max_height, 100)
    
    return max_width, max_height

def transform_perspective(
    image: np.ndarray,
    pts: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int], np.ndarray]:
    """
    Applies homography / perspective transformation to crop and unwarp document.
    
    Args:
        image: Original input image (BGR or Gray).
        pts: 4 corner coordinates (un-scaled to image resolution).
        
    Returns:
        tuple of:
        - warped_image: Perspective corrected rectangular document.
        - H: 3x3 homography / perspective transformation matrix.
        - (output_width, output_height)
        - ordered_pts: np.ndarray of shape (4, 2) containing ordered TL, TR, BR, BL.
    """
    ordered_pts = order_points(pts)
    max_width, max_height = compute_target_dimensions(ordered_pts)
    
    # Target destination rectangle coordinates
    dst_pts = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")
    
    # Calculate transformation matrix and warp perspective
    H = cv2.getPerspectiveTransform(ordered_pts, dst_pts)
    warped = cv2.warpPerspective(image, H, (max_width, max_height))
    
    return warped, H, (max_width, max_height), ordered_pts
