import cv2
import numpy as np

def enhance_original_corrected(warped_bgr: np.ndarray) -> np.ndarray:
    """
    Original Corrected Mode: Returns perspective corrected document with minimal processing,
    preserving natural white balance and color reproduction.
    """
    # Subtle brightness / white balance normalization
    ycrcb = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2YCrCb)
    channels = list(cv2.split(ycrcb))
    
    # Mild contrast normalization on Y channel
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
    channels[0] = clahe.apply(channels[0])
    
    normalized = cv2.merge(channels)
    return cv2.cvtColor(normalized, cv2.COLOR_YCrCb2BGR)

def enhance_color(warped_bgr: np.ndarray) -> np.ndarray:
    """
    Enhanced Color Mode: Performs LAB-space CLAHE contrast enhancement,
    bilateral background noise reduction, and unsharp masking for sharp, vivid documents.
    """
    # 1. Convert BGR to LAB color space
    lab = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    
    # 2. CLAHE on Luminance channel
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_chan)
    
    lab_enhanced = cv2.merge((l_enhanced, a_chan, b_chan))
    color_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    # 3. Bilateral Filter to smooth paper surface while keeping sharp edges
    filtered = cv2.bilateralFilter(color_enhanced, d=5, sigmaColor=40, sigmaSpace=40)
    
    # 4. Unsharp Masking for crisp text details
    blurred = cv2.GaussianBlur(filtered, (0, 0), 3.0)
    sharpened = cv2.addWeighted(filtered, 1.4, blurred, -0.4, 0)
    
    return sharpened

def enhance_scanner_bw(warped_bgr: np.ndarray) -> np.ndarray:
    """
    Scanner Mode B&W: Creates a clean scanner-style document using morphological background division
    to cancel uneven shadows and Gaussian adaptive thresholding for high contrast black text on white paper.
    """
    # 1. Grayscale
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY) if len(warped_bgr.shape) == 3 else warped_bgr.copy()
    
    # 2. Background Estimation via Morphological Closing
    kernel_dim = max(21, min(gray.shape) // 18)
    if kernel_dim % 2 == 0:
        kernel_dim += 1
    bg_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_dim, kernel_dim))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, bg_kernel)
    
    # 3. Illumination Division Normalization
    norm_float = np.divide(gray.astype(np.float32), np.maximum(background.astype(np.float32), 1.0)) * 255.0
    normalized = np.clip(norm_float, 0, 255).astype(np.uint8)
    
    # 4. Adaptive Gaussian Thresholding
    block_size = max(11, int(min(normalized.shape) / 30))
    if block_size % 2 == 0:
        block_size += 1
        
    binary = cv2.adaptiveThreshold(
        normalized,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=block_size,
        C=10
    )
    
    # 5. Morphological cleanup (median blur to clear speckles)
    cleaned = cv2.medianBlur(binary, 3)
    
    return cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)

def apply_enhancement(warped_bgr: np.ndarray, mode: str = "enhanced_color") -> np.ndarray:
    """
    Applies selected OpenCV enhancement pipeline.
    """
    mode_clean = mode.lower().strip()
    if mode_clean == "original_corrected":
        return enhance_original_corrected(warped_bgr)
    elif mode_clean == "scanner_bw":
        return enhance_scanner_bw(warped_bgr)
    else:
        return enhance_color(warped_bgr)
