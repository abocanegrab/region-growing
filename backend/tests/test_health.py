"""
Health check endpoint tests
"""
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "message" in data


def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "version" in data
    assert data["status"] == "running"


def test_sentinel_hub_test():
    """Test Sentinel Hub connection endpoint"""
    response = client.get("/api/analysis/test")
    
    assert response.status_code == 200
    data = response.json()
    assert "success" in data
    assert "message" in data


def test_openapi_docs():
    """Test that OpenAPI docs are accessible"""
    response = client.get("/api/docs")
    assert response.status_code == 200
    
    response = client.get("/api/redoc")
    assert response.status_code == 200


def test_openapi_schema():
    """Test that OpenAPI schema is valid"""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    
    schema = response.json()
    assert "openapi" in schema
    assert "info" in schema
    assert "paths" in schema
