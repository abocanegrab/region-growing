"""
API tests for embeddings endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np

from backend.app.main import app


client = TestClient(app)


class TestEmbeddingsExtractEndpoint:
    """Tests for POST /api/embeddings/extract endpoint"""
    
    @patch('backend.app.services.embeddings_service.EmbeddingsService')
    def test_extract_embeddings_success(self, mock_service_class):
        """Test successful embeddings extraction"""
        mock_service = MagicMock()
        mock_service.extract_embeddings_from_bbox.return_value = (
            np.random.rand(512, 512, 256),
            {'bbox': {}, 'date_from': '2024-01-01'}
        )
        mock_service.save_embeddings_to_disk.return_value = (
            'test-id-123',
            '/path/to/embeddings.npz'
        )
        mock_service_class.return_value = mock_service
        
        response = client.post(
            "/api/embeddings/extract",
            json={
                "bbox": {
                    "min_lat": 32.45,
                    "min_lon": -115.35,
                    "max_lat": 32.55,
                    "max_lon": -115.25
                },
                "use_simple_model": True
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data['success'] is True
        assert 'embeddings_shape' in data['data']
        assert 'embeddings_id' in data['data']
    
    def test_extract_embeddings_invalid_bbox(self):
        """Test extraction with invalid bbox"""
        response = client.post(
            "/api/embeddings/extract",
            json={
                "bbox": {
                    "min_lat": 100,
                    "min_lon": -115.35,
                    "max_lat": 32.55,
                    "max_lon": -115.25
                }
            }
        )
        assert response.status_code == 422


class TestEmbeddingsDownloadEndpoint:
    """Tests for GET /api/embeddings/download/{id} endpoint"""
    
    @patch('backend.app.services.embeddings_service.EmbeddingsService')
    def test_download_embeddings_success(self, mock_service_class):
        """Test successful embeddings download"""
        mock_service = MagicMock()
        mock_service.get_embeddings_file_path.return_value = MagicMock(exists=lambda: True)
        mock_service_class.return_value = mock_service
        
        response = client.get("/api/embeddings/download/test-id-123")
        assert response.status_code == 200
    
    @patch('backend.app.services.embeddings_service.EmbeddingsService')
    def test_download_embeddings_not_found(self, mock_service_class):
        """Test download of non-existent embeddings"""
        mock_service = MagicMock()
        mock_service.get_embeddings_file_path.return_value = MagicMock(exists=lambda: False)
        mock_service_class.return_value = mock_service
        
        response = client.get("/api/embeddings/download/nonexistent-id")
        assert response.status_code == 404


class TestEmbeddingsSimilarityEndpoint:
    """Tests for POST /api/embeddings/similarity endpoint"""
    
    @patch('backend.app.services.embeddings_service.EmbeddingsService')
    def test_compute_similarity_success(self, mock_service_class):
        """Test successful similarity computation"""
        mock_service = MagicMock()
        mock_service.compute_similarity_between_regions.return_value = {
            'mean_similarity': 0.87,
            'std_similarity': 0.12,
            'min_similarity': 0.45,
            'max_similarity': 0.99
        }
        mock_service_class.return_value = mock_service
        
        response = client.post(
            "/api/embeddings/similarity",
            json={
                "bbox_a": {
                    "min_lat": 32.45,
                    "min_lon": -115.35,
                    "max_lat": 32.55,
                    "max_lon": -115.25
                },
                "bbox_b": {
                    "min_lat": 20.85,
                    "min_lon": -101.45,
                    "max_lat": 20.95,
                    "max_lon": -101.35
                }
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data['success'] is True
        assert 'mean_similarity' in data['data']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
