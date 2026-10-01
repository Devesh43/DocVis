import os
import sys
import cv2
import numpy as np
import pytest

# Ensure backend package is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.scanner.pipeline import DocumentScanner

def generate_benchmark_test_set():
    """Generates 8 distinct test document images covering challenging real-world scenarios."""
    os.makedirs("benchmark_images", exist_ok=True)
    images = {}
    
    # Base Document Content
    def create_paper_content(w=400, h=550):
        paper = np.full((h, w, 3), (250, 248, 242), dtype=np.uint8)
        cv2.rectangle(paper, (20, 20), (w - 20, 70), (40, 70, 150), -1)
        cv2.putText(paper, "TEST DOCUMENT SCAN", (30, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        for i in range(10):
            y = 110 + i * 35
            cv2.putText(paper, f"Line item record {i+1}: Computer Vision Benchmarking Data", (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (50, 50, 50), 1)
            cv2.line(paper, (30, y + 8), (w - 30, y + 8), (220, 220, 220), 1)
        cv2.circle(paper, (w - 80, h - 80), 30, (180, 50, 50), 2)
        cv2.putText(paper, "PASSED", (w - 105, h - 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 50, 50), 2)
        return paper

    # 1. Straight-on photo on dark desk
    bg1 = np.full((800, 800, 3), (40, 42, 48), dtype=np.uint8)
    doc1 = create_paper_content(450, 600)
    bg1[100:700, 175:625] = doc1
    images["1_straight_on"] = bg1

    # 2. Angled photo on dark surface
    bg2 = np.full((800, 1000, 3), (35, 38, 45), dtype=np.uint8)
    doc2 = create_paper_content(450, 600)
    src2 = np.float32([[0,0], [450,0], [450,600], [0,600]])
    dst2 = np.float32([[180,100], [820,140], [750,720], [120,680]])
    M2 = cv2.getPerspectiveTransform(src2, dst2)
    warped2 = cv2.warpPerspective(doc2, M2, (1000, 800))
    mask2 = cv2.warpPerspective(np.full((600, 450), 255, dtype=np.uint8), M2, (1000, 800))
    img2 = cv2.add(cv2.bitwise_and(bg2, bg2, mask=cv2.bitwise_not(mask2)), warped2)
    images["2_angled"] = img2

    # 3. Dark background with texture
    bg3 = np.full((800, 800, 3), (20, 25, 30), dtype=np.uint8)
    noise3 = np.random.randint(-15, 15, bg3.shape, dtype=np.int16)
    bg3 = np.clip(bg3.astype(np.int16) + noise3, 0, 255).astype(np.uint8)
    doc3 = create_paper_content(420, 580)
    bg3[110:690, 190:610] = doc3
    images["3_dark_background"] = bg3

    # 4. Bright background (Light desk surface - low contrast border)
    bg4 = np.full((800, 800, 3), (220, 218, 212), dtype=np.uint8)
    doc4 = create_paper_content(450, 600)
    bg4[100:700, 175:625] = doc4
    images["4_bright_background"] = bg4

    # 5. Document with diagonal heavy shadow across frame
    img5 = bg1.copy()
    shadow_mask = np.zeros((800, 800), dtype=np.float32)
    pts_shadow = np.array([[0,0], [800, 300], [800, 800], [0, 500]], np.int32)
    cv2.fillPoly(shadow_mask, [pts_shadow], 0.5)
    cv2.GaussianBlur(shadow_mask, (51, 51), 0, dst=shadow_mask)
    for c in range(3):
        img5[:, :, c] = np.clip(img5[:, :, c].astype(np.float32) * (1.0 - shadow_mask * 0.6), 0, 255).astype(np.uint8)
    images["5_shadows"] = img5

    # 6. Partial frame document (occupying ~25% frame area)
    bg6 = np.full((900, 900, 3), (45, 45, 50), dtype=np.uint8)
    doc6 = create_paper_content(300, 420)
    bg6[240:660, 300:600] = doc6
    images["6_small_in_frame"] = bg6

    # 7. Rotated document (35 degrees tilt)
    bg7 = np.full((800, 800, 3), (40, 40, 40), dtype=np.uint8)
    doc7 = create_paper_content(380, 520)
    # Center rotation
    M7 = cv2.getRotationMatrix2D((190, 260), 35, 1.0)
    rotated_doc7 = cv2.warpAffine(doc7, M7, (380, 520), borderValue=(40, 40, 40))
    bg7[140:660, 210:590] = rotated_doc7
    images["7_rotated_35deg"] = bg7

    # 8. Cluttered background with multiple surrounding rectangles/boxes
    bg8 = bg1.copy()
    # Add surrounding clutter items
    cv2.rectangle(bg8, (50, 50), (140, 300), (90, 90, 100), -1) # Pen holder
    cv2.rectangle(bg8, (650, 450), (760, 750), (110, 80, 60), -1) # Notebook
    cv2.circle(bg8, (700, 150), 45, (200, 190, 180), -1) # Coffee mug
    images["8_cluttered_background"] = bg8

    for name, img in images.items():
        cv2.imwrite(f"benchmark_images/{name}.jpg", img)

    return images

def test_run_benchmark():
    images = generate_benchmark_test_set()
    scanner = DocumentScanner(max_dim=1000)
    
    results = {}
    print("\n" + "="*70)
    print("      DOCVIS COMPUTER VISION BENCHMARK SUITE - TEST REPORT")
    print("="*70)
    print(f"{'Scene Name':<25} | {'Detected?':<10} | {'Conf %':<8} | {'Mode':<10} | {'Time (ms)':<10}")
    print("-" * 70)
    
    passed_count = 0
    total_count = len(images)
    
    for name, img in images.items():
        res = scanner.process(img, mode="enhanced_color")
        is_auto = res["metrics"]["is_auto_detected"]
        conf = res["metrics"]["confidence_pct"]
        mode = res["metrics"]["corner_selection_mode"]
        t_ms = res["metrics"]["processing_time_ms"]
        
        status = "PASSED" if is_auto else "UNCERTAIN"
        if is_auto:
            passed_count += 1
            
        print(f"{name:<25} | {status:<10} | {conf:<8}% | {mode:<10} | {t_ms:<10} ms")
        results[name] = res
        
    print("-" * 70)
    print(f"AUTOMATIC DETECTION SUCCESS RATE: {passed_count}/{total_count} ({round((passed_count/total_count)*100, 1)}%)")
    print("="*70 + "\n")
    
    assert passed_count >= 6, "Detector should automatically detect at least 75% of benchmark test cases"

if __name__ == "__main__":
    test_run_benchmark()
