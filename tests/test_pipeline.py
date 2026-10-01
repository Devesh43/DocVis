import pytest
import cv2
import numpy as np
import os
import sys

# Ensure backend package is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.scanner.preprocessing import preprocess_image, resize_image, to_grayscale
from backend.scanner.perspective import order_points, compute_target_dimensions, transform_perspective
from backend.scanner.detection import detect_edges, find_document_contour
from backend.scanner.enhancement import apply_enhancement
from backend.scanner.pipeline import DocumentScanner

def test_order_points():
    # Unordered 4 corners
    pts = np.array([
        [300, 25],   # TR
        [10, 20],    # TL
        [15, 390],   # BL
        [290, 400]   # BR
    ], dtype=np.float32)
    
    ordered = order_points(pts)
    assert np.allclose(ordered[0], [10, 20])   # TL
    assert np.allclose(ordered[1], [300, 25])  # TR
    assert np.allclose(ordered[2], [290, 400]) # BR
    assert np.allclose(ordered[3], [15, 390])  # BL

def test_preprocessing():
    dummy = np.zeros((1200, 1600, 3), dtype=np.uint8)
    resized, scale = resize_image(dummy, max_dim=1000)
    assert max(resized.shape[:2]) == 1000
    assert scale == 1000 / 1600.0

def test_full_pipeline_on_sample():
    sample_path = "sample_images/invoice.jpg"
    assert os.path.exists(sample_path), "Sample image invoice.jpg should exist"
    
    image = cv2.imread(sample_path)
    assert image is not None
    
    scanner = DocumentScanner()
    result = scanner.process(image, mode="scanner_bw")
    
    assert "stages" in result
    assert "metrics" in result
    assert "cv_analysis" in result
    
    stages = result["stages"]
    for stage_key in ["original", "grayscale", "edge_map", "detected_contour", "perspective_corrected", "enhanced_result"]:
        assert stage_key in stages
        assert stages[stage_key].startswith("data:image/")
        
    metrics = result["metrics"]
    assert metrics["original_resolution"]["width"] > 0
    assert metrics["output_resolution"]["width"] > 0
    assert metrics["processing_time_ms"] > 0
    assert metrics["document_area_percent"] > 0

def test_manual_corner_override():
    sample_path = "sample_images/invoice.jpg"
    image = cv2.imread(sample_path)
    h, w = image.shape[:2]
    
    manual_corners = [[50, 50], [w - 50, 50], [w - 50, h - 50], [50, h - 50]]
    scanner = DocumentScanner()
    result = scanner.process(image, mode="enhanced_color", manual_corners=manual_corners)
    
    assert result["metrics"]["corner_selection_mode"] == "manual"
    assert result["metrics"]["is_auto_detected"] is False
