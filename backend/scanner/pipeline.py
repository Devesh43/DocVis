import cv2
import numpy as np
import time
import base64
from typing import Optional, List, Dict, Any, Tuple

from .preprocessing import preprocess_image, to_grayscale
from .detection import find_document_contour
from .perspective import transform_perspective, order_points
from .enhancement import apply_enhancement

def make_json_serializable(obj: Any) -> Any:
    """Recursively converts NumPy arrays, float32/int64 into native Python types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64, np.floating)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64, np.integer, np.bool_)):
        return int(obj) if not isinstance(obj, np.bool_) else bool(obj)
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    return obj

def encode_image_to_base64(img: np.ndarray, format: str = ".jpg", quality: int = 90) -> str:
    """Encodes a BGR or Gray numpy array to JPEG/PNG base64 data URI string."""
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if format in [".jpg", ".jpeg"] else []
    success, buffer = cv2.imencode(format, img, params)
    if not success:
        raise ValueError("Failed to encode image to base64 buffer")
    b64_str = base64.b64encode(buffer.tobytes()).decode("utf-8")
    mime = "image/jpeg" if format in [".jpg", ".jpeg"] else "image/png"
    return f"data:{mime};base64,{b64_str}"

def draw_contour_overlay(
    image: np.ndarray,
    ordered_corners: np.ndarray,
    is_auto: bool,
    confidence_pct: int,
    rejected_contours: List[np.ndarray] = None
) -> np.ndarray:
    """
    Draws the detected document overlay:
    - Rejected major contour candidates drawn in thin gray lines
    - Selected document quadrilateral contour drawn prominently
    - 4 labeled corner markers (TL, TR, BR, BL)
    - Confidence label tag
    """
    overlay = image.copy()
    
    # 1. Draw rejected candidate contours in thin gray/red lines
    if rejected_contours:
        for rcnt in rejected_contours:
            cv2.drawContours(overlay, [rcnt], -1, (100, 100, 100), 1)
            
    # 2. Draw selected document polygon
    pts = ordered_corners.astype(np.int32).reshape((-1, 1, 2))
    poly_color = (0, 230, 115) if is_auto else (0, 180, 255)
    cv2.polylines(overlay, [pts], isClosed=True, color=poly_color, thickness=3)
    
    fill = overlay.copy()
    cv2.fillPoly(fill, [pts], poly_color)
    cv2.addWeighted(fill, 0.15, overlay, 0.85, 0, overlay)
    
    # 3. Draw Corner handles and coordinate labels
    labels = ["TL", "TR", "BR", "BL"]
    corner_colors = [(0, 0, 255), (0, 255, 0), (255, 128, 0), (255, 0, 255)]
    
    for i, (x, y) in enumerate(ordered_corners):
        ix, iy = int(round(x)), int(round(y))
        cv2.circle(overlay, (ix, iy), 8, corner_colors[i], -1)
        cv2.circle(overlay, (ix, iy), 11, (255, 255, 255), 2)
        
        lbl = f"{labels[i]} ({ix},{iy})"
        cv2.putText(overlay, lbl, (ix + 12, iy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(overlay, lbl, (ix + 12, iy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
    # 4. Confidence Score Tag
    tag_text = f"Confidence: {confidence_pct}% ({'AUTO' if is_auto else 'MANUAL/FALLBACK'})"
    cv2.putText(overlay, tag_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
    cv2.putText(overlay, tag_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200) if is_auto else (0, 200, 255), 1)
    
    return overlay

class DocumentScanner:
    """
    Main Document Vision Pipeline Manager.
    Executes full processing flow and collects intermediate outputs & CV metrics.
    """
    def __init__(self, max_dim: int = 1000):
        self.max_dim = max_dim

    def process(
        self,
        image: np.ndarray,
        mode: str = "enhanced_color",
        manual_corners: Optional[List[List[float]]] = None,
        canny_low: Optional[int] = 50,
        canny_high: Optional[int] = 150
    ) -> Dict[str, Any]:
        """
        Runs complete computer vision document scanning pipeline.
        """
        start_total = time.perf_counter()
        timings = {}
        
        orig_h, orig_w = image.shape[:2]
        
        # 1. Preprocessing
        t0 = time.perf_counter()
        resized_color, gray, gray_blur, scale_factor = preprocess_image(image, max_dim=self.max_dim)
        timings["preprocessing_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        
        # 2 & 3. Edge Detection & Multi-Pass Contour Search
        t0 = time.perf_counter()
        
        if manual_corners is not None and len(manual_corners) == 4:
            # User provided manual corners in original image resolution -> scale to resized
            scaled_manual = np.array(manual_corners, dtype=np.float32) * float(scale_factor)
            ordered_resized_corners = order_points(scaled_manual)
            is_auto = False
            confidence_pct = 100 # User confirmed
            confidence_label = "Manual Selected"
            status_msg = "Manual corner override applied."
            contours_evaluated = []
            quadrilaterals_found = 0
            rejected_contours = []
            # Generate primary edge map for Edge Map stage display
            _, _, _, _ = preprocess_image(image, max_dim=self.max_dim)
            from .detection import generate_multi_edge_maps
            edge_maps = generate_multi_edge_maps(gray_blur, canny_low, canny_high)
            edge_map = edge_maps[0][1]
        else:
            detect_res = find_document_contour(
                gray_blur,
                image_shape=resized_color.shape,
                canny_low=canny_low,
                canny_high=canny_high
            )
            ordered_resized_corners = detect_res["selected_corners"]
            is_auto = detect_res["is_auto_detected"]
            confidence_pct = detect_res["confidence_pct"]
            confidence_label = detect_res["confidence_label"]
            status_msg = detect_res["status_message"]
            edge_map = detect_res["primary_edge_map"]
            contours_evaluated = detect_res["contours_evaluated"]
            quadrilaterals_found = detect_res["quadrilaterals_found"]
            rejected_contours = detect_res["rejected_contours"]
            
        timings["edge_and_detection_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        
        # Unscale corner coordinates back to original image dimensions
        original_corners = ordered_resized_corners / float(scale_factor)
        
        # 4. Contour Visualization (Stage 4)
        contour_viz = draw_contour_overlay(
            resized_color,
            ordered_resized_corners,
            is_auto=is_auto,
            confidence_pct=confidence_pct,
            rejected_contours=rejected_contours
        )
        
        # 5. Perspective Transformation (Warping on full-res image)
        t0 = time.perf_counter()
        warped, transform_matrix, (out_w, out_h), ordered_orig_corners = transform_perspective(
            image,
            original_corners
        )
        timings["perspective_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        
        # 6. Enhancement Pipeline Execution
        t0 = time.perf_counter()
        enhanced = apply_enhancement(warped, mode=mode)
        timings["enhancement_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        
        total_time_ms = round((time.perf_counter() - start_total) * 1000, 2)
        timings["total_ms"] = total_time_ms
        
        # Document Area Percentage (polygon area relative to original frame area)
        x = ordered_orig_corners[:, 0]
        y = ordered_orig_corners[:, 1]
        poly_area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        doc_area_pct = round((poly_area / float(orig_h * orig_w)) * 100.0, 2)
        
        # Format Intermediate Stage Outputs
        edge_bgr = cv2.cvtColor(edge_map, cv2.COLOR_GRAY2BGR)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        stages = {
            "original": encode_image_to_base64(resized_color),
            "grayscale": encode_image_to_base64(gray_bgr),
            "edge_map": encode_image_to_base64(edge_bgr),
            "detected_contour": encode_image_to_base64(contour_viz),
            "perspective_corrected": encode_image_to_base64(warped),
            "enhanced_result": encode_image_to_base64(enhanced)
        }
        
        metrics = {
            "original_resolution": {"width": int(orig_w), "height": int(orig_h)},
            "output_resolution": {"width": int(out_w), "height": int(out_h)},
            "processing_time_ms": float(total_time_ms),
            "timings_breakdown_ms": timings,
            "document_area_percent": float(doc_area_pct),
            "is_auto_detected": bool(is_auto),
            "confidence_pct": int(confidence_pct),
            "confidence_label": confidence_label,
            "status_message": status_msg,
            "corner_selection_mode": "auto" if is_auto else ("manual" if manual_corners else "fallback")
        }
        
        cv_analysis = {
            "canny_thresholds": {"low": canny_low, "high": canny_high},
            "scale_factor": round(float(scale_factor), 4),
            "corners_original": ordered_orig_corners.round(1).tolist(),
            "corners_resized": ordered_resized_corners.round(1).tolist(),
            "homography_matrix": transform_matrix.tolist(),
            "contours_evaluated": contours_evaluated,
            "quadrilaterals_found": quadrilaterals_found
        }
        
        result_dict = {
            "stages": stages,
            "metrics": metrics,
            "cv_analysis": cv_analysis
        }
        
        return make_json_serializable(result_dict)
