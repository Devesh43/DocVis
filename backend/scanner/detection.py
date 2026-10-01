import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from .perspective import order_points

def compute_interior_angles(pts: np.ndarray) -> List[float]:
    """Calculates interior angles in degrees for a 4-point polygon ordered [TL, TR, BR, BL]."""
    angles = []
    for i in range(4):
        p0 = pts[i]
        p1 = pts[(i + 1) % 4]
        p2 = pts[(i - 1) % 4]
        
        v1 = p1 - p0
        v2 = p2 - p0
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            angles.append(0.0)
            continue
            
        cos_theta = np.dot(v1, v2) / (norm1 * norm2)
        angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
        angles.append(float(angle))
    return angles

def score_quadrilateral(quad_pts: np.ndarray, frame_shape: Tuple[int, int]) -> Dict[str, Any]:
    """
    Evaluates a candidate 4-point polygon and returns a confidence score (0.0 to 1.0).
    """
    h, w = frame_shape[:2]
    frame_area = float(h * w)
    
    pts = quad_pts.reshape((4, 2))
    area = float(cv2.contourArea(pts))
    area_ratio = area / frame_area
    
    if area_ratio < 0.04 or area_ratio > 0.96:
        area_score = 0.0
    else:
        # Prefer documents occupying 15% to 85% of frame
        area_score = min(1.0, area_ratio / 0.25)
        
    # Rectangularity score (contour area vs area of minimum bounding rotated box)
    rect = cv2.minAreaRect(pts)
    rect_area = float(rect[1][0] * rect[1][1])
    rect_score = (area / rect_area) if rect_area > 0 else 0.0
    rect_score = min(1.0, rect_score)
    
    # Convexity score
    is_convex = bool(cv2.isContourConvex(pts.reshape((-1, 1, 2))))
    convex_score = 1.0 if is_convex else 0.3
    
    # Angle score (interior angles near 90 degrees)
    ordered = order_points(pts)
    angles = compute_interior_angles(ordered)
    angle_deviations = [abs(ang - 90.0) for ang in angles]
    max_dev = max(angle_deviations) if angles else 90.0
    
    if max_dev > 55:
        angle_score = 0.1
    else:
        angle_score = max(0.0, 1.0 - (max_dev / 50.0))
        
    # Aspect Ratio sanity check
    bw, bh = rect[1]
    aspect_ratio = float(bw / bh) if bh > 0 else 1.0
    if aspect_ratio < 0.2 or aspect_ratio > 5.0:
        aspect_score = 0.2
    else:
        aspect_score = 1.0
        
    total_score = (
        0.35 * area_score +
        0.30 * rect_score +
        0.15 * convex_score +
        0.12 * angle_score +
        0.08 * aspect_score
    )
    
    confidence_pct = int(round(total_score * 100))
    
    return {
        "score": round(float(total_score), 4),
        "confidence_pct": confidence_pct,
        "area": round(area, 1),
        "area_percentage": round(area_ratio * 100.0, 2),
        "rectangularity": round(rect_score, 2),
        "is_convex": is_convex,
        "angles": [round(a, 1) for a in angles],
        "aspect_ratio": round(aspect_ratio, 2)
    }

def detect_edges(
    gray_blur: np.ndarray,
    low_threshold: Optional[int] = 50,
    high_threshold: Optional[int] = 150,
    apply_morphology: bool = True
) -> np.ndarray:
    """Convenience wrapper for single Canny edge detection."""
    low = low_threshold if low_threshold is not None else 50
    high = high_threshold if high_threshold is not None else 150
    edges = cv2.Canny(gray_blur, low, high)
    if apply_morphology:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        edges = cv2.dilate(edges, kernel, iterations=1)
    return edges

def generate_multi_edge_maps(
    gray_blur: np.ndarray,
    canny_low: Optional[int] = 50,
    canny_high: Optional[int] = 150
) -> List[Tuple[str, np.ndarray]]:
    """
    Generates multiple edge maps using complementary vision operations
    to ensure reliable document boundary localization across different lighting/contrast scenes.
    """
    edge_maps = []
    
    # Ensure valid Canny thresholds
    low = canny_low if canny_low is not None else 50
    high = canny_high if canny_high is not None else 150
    if low >= high:
        low = max(10, high - 30)
        
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    
    # Pass 1: Primary User-Configured / Auto Canny
    edges1 = cv2.Canny(gray_blur, low, high)
    edges1 = cv2.morphologyEx(edges1, cv2.MORPH_CLOSE, kernel_close)
    edges1 = cv2.dilate(edges1, kernel_close, iterations=1)
    edge_maps.append(("Standard Canny", edges1))
    
    # Pass 2: CLAHE Contrast Enhanced Canny (for low contrast paper on bright surface)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray_blur)
    edges2 = cv2.Canny(gray_clahe, max(20, low - 20), min(220, high + 20))
    edges2 = cv2.morphologyEx(edges2, cv2.MORPH_CLOSE, kernel_close)
    edges2 = cv2.dilate(edges2, kernel_close, iterations=1)
    edge_maps.append(("CLAHE Canny", edges2))
    
    # Pass 3: Adaptive Gaussian Morphological Edge Map (for shadowed paper)
    adaptive_thresh = cv2.adaptiveThreshold(
        gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    edges3 = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel_close)
    edge_maps.append(("Adaptive Morphology", edges3))
    
    # Pass 4: Otsu Binarization Morphological Boundary
    _, otsu = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    edges4 = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel_close)
    edges4 = cv2.Canny(edges4, 50, 150)
    edge_maps.append(("Otsu Edge Map", edges4))
    
    return edge_maps

def find_document_contour(
    gray_blur: np.ndarray,
    image_shape: Tuple[int, int],
    canny_low: Optional[int] = 50,
    canny_high: Optional[int] = 150,
    min_confidence: float = 40.0
) -> Dict[str, Any]:
    """
    Searches for document boundary using multi-pass edge extraction and heuristic scoring.
    """
    height, width = image_shape[:2]
    total_area = height * width
    
    edge_maps = generate_multi_edge_maps(gray_blur, canny_low, canny_high)
    primary_edge_map = edge_maps[0][1]
    
    all_candidates = []
    rejected_contours = []
    seen_quad_hashes = set()
    
    for pass_name, emap in edge_maps:
        contours, _ = cv2.findContours(emap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        for cnt in sorted_contours[:12]:
            area = float(cv2.contourArea(cnt))
            if area < total_area * 0.03:
                continue
                
            peri = float(cv2.arcLength(cnt, True))
            if peri == 0:
                continue
                
            quad_approx = None
            for eps_factor in [0.015, 0.02, 0.025, 0.03, 0.04]:
                approx = cv2.approxPolyDP(cnt, eps_factor * peri, True)
                if len(approx) == 4:
                    quad_approx = approx
                    break
                    
            if quad_approx is None:
                hull = cv2.convexHull(cnt)
                hull_peri = float(cv2.arcLength(hull, True))
                if hull_peri > 0:
                    for eps_factor in [0.02, 0.03, 0.04]:
                        approx = cv2.approxPolyDP(hull, eps_factor * hull_peri, True)
                        if len(approx) == 4:
                            quad_approx = approx
                            break
                            
            if quad_approx is not None:
                pts = quad_approx.reshape((4, 2))
                ordered = order_points(pts)
                quad_key = tuple(np.round(ordered / 15.0).astype(int).flatten())
                if quad_key in seen_quad_hashes:
                    continue
                seen_quad_hashes.add(quad_key)
                
                score_info = score_quadrilateral(pts, image_shape)
                score_info["pass_source"] = pass_name
                score_info["contour_pts"] = ordered
                score_info["raw_contour"] = cnt
                all_candidates.append(score_info)
            else:
                if len(rejected_contours) < 10:
                    rejected_contours.append(cnt)

    all_candidates.sort(key=lambda c: c["score"], reverse=True)
    
    evaluated_meta = []
    for idx, cand in enumerate(all_candidates[:10]):
        evaluated_meta.append({
            "index": idx,
            "pass_source": cand["pass_source"],
            "area_percentage": cand["area_percentage"],
            "confidence_pct": cand["confidence_pct"],
            "rectangularity": cand["rectangularity"],
            "is_convex": cand["is_convex"],
            "selected": False
        })
        
    best_candidate = all_candidates[0] if all_candidates else None
    
    if best_candidate and best_candidate["confidence_pct"] >= min_confidence:
        best_candidate_corners = best_candidate["contour_pts"]
        is_auto = True
        conf_pct = best_candidate["confidence_pct"]
        conf_label = "High Confidence" if conf_pct >= 70 else "Moderate Confidence"
        msg = f"Document detected ({conf_label} {conf_pct}%)"
        if evaluated_meta:
            evaluated_meta[0]["selected"] = True
    else:
        margin_w = int(width * 0.10)
        margin_h = int(height * 0.10)
        fallback_corners = np.array([
            [margin_w, margin_h],
            [width - margin_w, margin_h],
            [width - margin_w, height - margin_h],
            [margin_w, height - margin_h]
        ], dtype=np.float32)
        
        best_candidate_corners = order_points(fallback_corners)
        is_auto = False
        conf_pct = best_candidate["confidence_pct"] if best_candidate else 20
        conf_label = "Low / Uncertain"
        msg = "Automatic document detection uncertain — adjust corners manually."
        
    return {
        "selected_corners": best_candidate_corners,
        "is_auto_detected": is_auto,
        "confidence_pct": conf_pct,
        "confidence_label": conf_label,
        "status_message": msg,
        "primary_edge_map": primary_edge_map,
        "contours_evaluated": evaluated_meta,
        "quadrilaterals_found": len(all_candidates),
        "rejected_contours": rejected_contours[:6]
    }
