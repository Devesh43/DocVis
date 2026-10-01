import cv2
import numpy as np

def resize_image(image: np.ndarray, max_dim: int = 1000) -> tuple[np.ndarray, float]:
    """
    Resizes an image maintaining aspect ratio so its largest dimension is max_dim.
    
    Args:
        image: Input image as numpy array (H, W, C) or (H, W).
        max_dim: Target maximum dimension (height or width).
        
    Returns:
        tuple of (resized_image, scale_factor)
        scale_factor = resized_dim / original_dim
    """
    height, width = image.shape[:2]
    max_current = max(height, width)
    
    if max_current <= max_dim:
        return image.copy(), 1.0
        
    scale_factor = max_dim / float(max_current)
    new_width = int(round(width * scale_factor))
    new_height = int(round(height * scale_factor))
    
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized, scale_factor

def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Converts a BGR image to Grayscale."""
    if len(image.shape) == 2:
        return image.copy()
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def apply_gaussian_blur(gray_image: np.ndarray, kernel_size: tuple[int, int] = (5, 5), sigma: float = 0.0) -> np.ndarray:
    """Applies Gaussian Blur to smooth out noise prior to edge detection."""
    # Ensure kernel size dimensions are odd numbers
    kx = kernel_size[0] if kernel_size[0] % 2 == 1 else kernel_size[0] + 1
    ky = kernel_size[1] if kernel_size[1] % 2 == 1 else kernel_size[1] + 1
    return cv2.GaussianBlur(gray_image, (kx, ky), sigmaX=sigma, sigmaY=sigma)

def preprocess_image(image: np.ndarray, max_dim: int = 1000, blur_kernel: tuple[int, int] = (5, 5)) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Full preprocessing step:
    1. Resize for consistent processing speed & edge scale.
    2. Grayscale conversion.
    3. Gaussian blur.
    
    Returns:
        (resized_color, resized_gray, blurred_gray, scale_factor)
    """
    resized_color, scale_factor = resize_image(image, max_dim=max_dim)
    gray = to_grayscale(resized_color)
    blurred = apply_gaussian_blur(gray, kernel_size=blur_kernel)
    return resized_color, gray, blurred, scale_factor
