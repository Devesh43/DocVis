import cv2
import numpy as np
import os

def create_sample_documents():
    os.makedirs("sample_images", exist_ok=True)
    
    # 1. Invoice Document (Angled on dark background)
    bg = np.full((800, 1000, 3), (40, 42, 48), dtype=np.uint8) # Dark desk surface
    # Add subtle texture / noise to background
    noise = np.random.randint(-10, 10, bg.shape, dtype=np.int16)
    bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Create flat document
    doc = np.full((600, 450, 3), (250, 248, 242), dtype=np.uint8) # Slightly off-white paper
    
    # Add document text & graphics
    cv2.rectangle(doc, (30, 30), (420, 90), (30, 60, 140), -1)
    cv2.putText(doc, "TECHCORP INVOICE", (45, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    
    cv2.putText(doc, "Invoice #: INV-2026-0891", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1)
    cv2.putText(doc, "Date: October 1, 2026", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1)
    cv2.putText(doc, "Billed To: Antigravity Systems Inc.", (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1)
    
    cv2.line(doc, (30, 190), (420, 190), (180, 180, 180), 2)
    
    # Table Header
    cv2.rectangle(doc, (30, 210), (420, 235), (220, 225, 215), -1)
    cv2.putText(doc, "Item Description", (40, 228), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1)
    cv2.putText(doc, "Qty", (260, 228), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1)
    cv2.putText(doc, "Rate", (320, 228), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1)
    cv2.putText(doc, "Total", (370, 228), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1)
    
    # Table Rows
    rows = [
        ("Computer Vision SDK", "1", "$1,200.00", "$1,200.00"),
        ("Image Processing Pipeline", "2", "$850.00", "$1,700.00"),
        ("Perspective Geometry Engine", "1", "$650.00", "$650.00"),
        ("Cloud Deployment Support", "5 hrs", "$120.00", "$600.00")
    ]
    
    y = 260
    for item, qty, rate, total in rows:
        cv2.putText(doc, item, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 50, 50), 1)
        cv2.putText(doc, qty, (265, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 50, 50), 1)
        cv2.putText(doc, rate, (320, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 50, 50), 1)
        cv2.putText(doc, total, (370, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 50, 50), 1)
        cv2.line(doc, (30, y + 8), (420, y + 8), (230, 230, 230), 1)
        y += 30
        
    cv2.line(doc, (30, 420), (420, 420), (40, 40, 40), 2)
    cv2.putText(doc, "GRAND TOTAL:  $4,150.00", (230, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2)
    
    # Add stamp/signature
    cv2.circle(doc, (100, 500), 35, (180, 50, 50), 2)
    cv2.putText(doc, "PAID", (80, 507), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 50, 50), 2)
    
    # Apply Perspective Transformation to warp paper into background
    src_pts = np.float32([[0, 0], [450, 0], [450, 600], [0, 600]])
    dst_pts = np.float32([[180, 100], [820, 140], [750, 720], [120, 680]]) # Perspective tilt
    
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped_doc = cv2.warpPerspective(doc, matrix, (1000, 800))
    
    # Mask background and merge
    mask = cv2.warpPerspective(np.full((600, 450), 255, dtype=np.uint8), matrix, (1000, 800))
    inv_mask = cv2.bitwise_not(mask)
    
    bg_masked = cv2.bitwise_and(bg, bg, mask=inv_mask)
    final_img = cv2.add(bg_masked, warped_doc)
    
    cv2.imwrite("sample_images/invoice.jpg", final_img)
    
    # 2. Receipt (tilted on blue tabletop)
    bg2 = np.full((800, 800, 3), (35, 60, 90), dtype=np.uint8) # Dark blue tabletop
    doc2 = np.full((550, 320, 3), (245, 245, 240), dtype=np.uint8)
    cv2.putText(doc2, "SUPERMARKET SCAN", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 10, 10), 2)
    cv2.putText(doc2, "-----------------------------", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    items = [
        ("Organic Milk 1L", "$3.49"),
        ("Whole Wheat Bread", "$2.99"),
        ("Fresh Apples 1kg", "$4.50"),
        ("Greek Yogurt 500g", "$5.20"),
        ("Espresso Beans 250g", "$8.90"),
        ("Dark Chocolate 85%", "$3.75"),
    ]
    y = 90
    for it, pr in items:
        cv2.putText(doc2, it, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (30, 30, 30), 1)
        cv2.putText(doc2, pr, (240, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (30, 30, 30), 1)
        y += 28
    cv2.putText(doc2, "-----------------------------", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    cv2.putText(doc2, "TOTAL: $28.83", (140, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    cv2.putText(doc2, "THANK YOU FOR YOUR VISIT!", (40, y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)
    
    src_pts2 = np.float32([[0, 0], [320, 0], [320, 550], [0, 550]])
    dst_pts2 = np.float32([[220, 120], [620, 80], [580, 710], [150, 660]])
    m2 = cv2.getPerspectiveTransform(src_pts2, dst_pts2)
    warped_doc2 = cv2.warpPerspective(doc2, m2, (800, 800))
    mask2 = cv2.warpPerspective(np.full((550, 320), 255, dtype=np.uint8), m2, (800, 800))
    bg2_masked = cv2.bitwise_and(bg2, bg2, mask=cv2.bitwise_not(mask2))
    img2 = cv2.add(bg2_masked, warped_doc2)
    cv2.imwrite("sample_images/receipt.jpg", img2)

    # 3. Document Page with diagram
    bg3 = np.full((900, 900, 3), (60, 50, 45), dtype=np.uint8) # Wooden desk
    doc3 = np.full((650, 480, 3), (252, 250, 245), dtype=np.uint8)
    cv2.putText(doc3, "RESEARCH PAPER SUMMARY", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2)
    cv2.putText(doc3, "Abstract: This document details perspective transformation...", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1)
    
    # Draw simple diagram box inside document
    cv2.rectangle(doc3, (60, 130), (420, 350), (230, 230, 240), -1)
    cv2.rectangle(doc3, (60, 130), (420, 350), (100, 100, 150), 2)
    cv2.rectangle(doc3, (90, 160), (220, 220), (100, 150, 220), -1)
    cv2.putText(doc3, "Input", (125, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.arrowedLine(doc3, (220, 190), (260, 190), (80, 80, 80), 2)
    cv2.rectangle(doc3, (260, 160), (390, 220), (100, 200, 150), -1)
    cv2.putText(doc3, "Transform", (280, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(doc3, "Figure 1: Homography Pipeline", (120, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)
    
    cv2.putText(doc3, "Conclusions:", (30, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 2)
    cv2.putText(doc3, "1. Homography allows mapping between 2D planes.", (30, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 50, 50), 1)
    cv2.putText(doc3, "2. Adaptive thresholding removes non-uniform shadows.", (30, 445), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 50, 50), 1)
    
    src_pts3 = np.float32([[0, 0], [480, 0], [480, 650], [0, 650]])
    dst_pts3 = np.float32([[140, 130], [780, 110], [720, 810], [100, 770]])
    m3 = cv2.getPerspectiveTransform(src_pts3, dst_pts3)
    warped_doc3 = cv2.warpPerspective(doc3, m3, (900, 900))
    mask3 = cv2.warpPerspective(np.full((650, 480), 255, dtype=np.uint8), m3, (900, 900))
    bg3_masked = cv2.bitwise_and(bg3, bg3, mask=cv2.bitwise_not(mask3))
    img3 = cv2.add(bg3_masked, warped_doc3)
    cv2.imwrite("sample_images/book_page.jpg", img3)

    print("Sample images created successfully!")

if __name__ == "__main__":
    create_sample_documents()
