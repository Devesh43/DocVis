import pytest
import os
import sys
from fastapi.testclient import TestClient

# Ensure backend package is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "OpenCV" in data["cv_engine"]

def test_get_samples():
    response = client.get("/api/samples")
    assert response.status_code == 200
    samples = response.json()
    assert len(samples) >= 3
    sample_ids = [s["id"] for s in samples]
    assert "invoice" in sample_ids
    assert "receipt" in sample_ids

def test_scan_sample():
    response = client.post(
        "/api/scan-sample/invoice",
        data={"mode": "scanner_bw"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "stages" in data
    assert "metrics" in data
    assert data["metrics"]["processing_time_ms"] > 0

def test_scan_upload():
    sample_file_path = "sample_images/receipt.jpg"
    assert os.path.exists(sample_file_path)
    
    with open(sample_file_path, "rb") as f:
        response = client.post(
            "/api/scan",
            files={"file": ("receipt.jpg", f, "image/jpeg")},
            data={"mode": "enhanced_color"}
        )
        
    assert response.status_code == 200
    data = response.json()
    assert data["stages"]["enhanced_result"].startswith("data:image/")
