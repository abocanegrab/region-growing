"""
Analysis endpoint tests
"""
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_analyze_endpoint_missing_bbox():
    """Test analyze endpoint without bbox (should fail validation)"""
    response = client.post("/api/analysis/analyze", json={})
    
    assert response.status_code == 422  # Validation error
    data = response.json()
    assert "detail" in data


def test_analyze_endpoint_invalid_bbox():
    """Test analyze endpoint with invalid bbox coordinates"""
    # max_lat < min_lat should fail validation
    response = client.post("/api/analysis/analyze", json={
        "bbox": {
            "min_lat": -11.95,
            "min_lon": -77.05,
            "max_lat": -12.05,  # Less than min_lat - should fail
            "max_lon": -76.95
        }
    })
    
    assert response.status_code == 422  # Validation error


def test_analyze_endpoint_valid_request_structure(monkeypatch):
    """Test analyze endpoint with a valid request, mocking the service layer."""
    
    # 1. Define a fake result that the service would normally return
    fake_service_result = {
        "geojson": {"type": "FeatureCollection", "features": []},
        "statistics": {"total_area": 1.0, "mean_ndvi": 0.5},
        "images": {"rgb": "fake_base64_string"}
    }
    
    # 2. Use monkeypatch to replace the real, slow method with a lambda
    #    that returns the fake result instantly.
    monkeypatch.setattr(
        "app.services.region_growing_service.RegionGrowingService.analyze_stress",
        lambda self, bbox, date_from, date_to: fake_service_result
    )
    
    # 3. Call the endpoint. It will now use the mocked method.
    response = client.post("/api/analysis/analyze", json={
        "bbox": {
            "min_lat": -12.05,
            "min_lon": -77.05,
            "max_lat": -11.95,
            "max_lon": -76.95
        },
        "date_from": "2024-01-01",
        "date_to": "2024-01-31",
        "method": "classic"
    })
    
    # 4. Assert that the request was successful and the data is correct.
    #    The test is now fast and reliable.
    assert response.status_code == 200
    response_data = response.json()
    assert response_data["success"] is True
    assert response_data["data"] == fake_service_result


def test_analyze_endpoint_invalid_method():
    """Test analyze endpoint with invalid method"""
    response = client.post("/api/analysis/analyze", json={
        "bbox": {
            "min_lat": -12.05,
            "min_lon": -77.05,
            "max_lat": -11.95,
            "max_lon": -76.95
        },
        "method": "invalid_method"  # Should fail validation
    })
    
    assert response.status_code == 422  # Validation error


def test_analyze_endpoint_invalid_date_format():
    """Test analyze endpoint with invalid date format"""
    response = client.post("/api/analysis/analyze", json={
        "bbox": {
            "min_lat": -12.05,
            "min_lon": -77.05,
            "max_lat": -11.95,
            "max_lon": -76.95
        },
        "date_from": "01-01-2024"  # Wrong format - should fail
    })
    
    assert response.status_code == 422  # Validation error


def test_analyze_endpoint_response_structure():
    """Test that successful responses have correct structure"""
    response = client.post("/api/analysis/analyze", json={
        "bbox": {
            "min_lat": -12.05,
            "min_lon": -77.05,
            "max_lat": -12.04,
            "max_lon": -77.04
        }
    })
    
    data = response.json()
    assert "success" in data
    
    # If successful, check data structure
    if data.get("success"):
        assert "data" in data
        # Additional checks could be added here


def test_analyze_endpoint_bbox_validation():
    """Test bbox field validation"""
    # Test latitude out of range
    response = client.post("/api/analysis/analyze", json={
        "bbox": {
            "min_lat": -95,  # Out of range
            "min_lon": -77.05,
            "max_lat": -11.95,
            "max_lon": -76.95
        }
    })
    assert response.status_code == 422
    
    # Test longitude out of range
    response = client.post("/api/analysis/analyze", json={
        "bbox": {
            "min_lat": -12.05,
            "min_lon": -200,  # Out of range
            "max_lat": -11.95,
            "max_lon": -76.95
        }
    })
    assert response.status_code == 422
