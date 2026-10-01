import os
import json
import numpy as np
import cv2
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from backend.scanner.pipeline import DocumentScanner

app = FastAPI(
    title="DocVis - Intelligent Document Vision System",
    description="Computer Vision backend using OpenCV for automated document boundary detection, perspective transformation, and image enhancement.",
    version="1.0.0"
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

scanner = DocumentScanner(max_dim=1000)

SAMPLE_IMAGES = {
    "invoice": {
        "id": "invoice",
        "title": "Angled Corporate Invoice",
        "filename": "invoice.jpg",
        "description": "Invoice photographed on dark surface with perspective tilt."
    },
    "receipt": {
        "id": "receipt",
        "title": "Supermarket Receipt",
        "filename": "receipt.jpg",
        "description": "Narrow store receipt with thermal text on tabletop."
    },
    "book_page": {
        "id": "book_page",
        "title": "Research Paper / Book Page",
        "filename": "book_page.jpg",
        "description": "Document page with figures and text at oblique angle."
    }
}

@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "system": "DocVis Intelligent Document Vision System",
        "cv_engine": "OpenCV " + cv2.__version__
    }

@app.get("/api/samples")
def get_samples():
    """Returns list of built-in sample document images."""
    return list(SAMPLE_IMAGES.values())

@app.post("/api/scan")
async def scan_document(
    file: UploadFile = File(...),
    mode: str = Form("enhanced_color"),
    manual_corners: Optional[str] = Form(None),
    canny_low: Optional[int] = Form(50),
    canny_high: Optional[int] = Form(150)
):
    """
    Processes an uploaded document image through the computer vision pipeline.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
        
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Failed to decode image data.")
            
        parsed_corners = None
        if manual_corners:
            try:
                parsed_corners = json.loads(manual_corners)
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid JSON for manual_corners.")
                
        result = scanner.process(
            image=image,
            mode=mode,
            manual_corners=parsed_corners,
            canny_low=canny_low,
            canny_high=canny_high
        )
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document vision processing error: {str(e)}")

@app.post("/api/scan-sample/{sample_id}")
def scan_sample(
    sample_id: str,
    mode: str = Form("enhanced_color"),
    manual_corners: Optional[str] = Form(None),
    canny_low: Optional[int] = Form(50),
    canny_high: Optional[int] = Form(150)
):
    """
    Processes a built-in sample image through the vision pipeline.
    """
    if sample_id not in SAMPLE_IMAGES:
        raise HTTPException(status_code=404, detail=f"Sample '{sample_id}' not found.")
        
    file_path = os.path.join("sample_images", SAMPLE_IMAGES[sample_id]["filename"])
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Sample file {file_path} missing.")
        
    image = cv2.imread(file_path)
    if image is None:
        raise HTTPException(status_code=500, detail="Failed to read sample image from disk.")
        
    parsed_corners = None
    if manual_corners:
        try:
            parsed_corners = json.loads(manual_corners)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON for manual_corners.")
            
    result = scanner.process(
        image=image,
        mode=mode,
        manual_corners=parsed_corners,
        canny_low=canny_low,
        canny_high=canny_high
    )
    return JSONResponse(content=result)

# Mount frontend directory for static UI serving
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

@app.get("/")
def serve_index():
    index_file = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "DocVis Backend API active. Frontend index.html not found."}
