"""
Integration tests for hierarchical analysis pipeline.

Tests the complete end-to-end pipeline from Sentinel-2 download to
classified stress maps using mocked external dependencies.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.pipeline.hierarchical_analysis import (
    HierarchicalAnalysisPipeline,
    AnalysisConfig,
    AnalysisResult
)


class TestHierarchicalPipelineValidation:
    """Test configuration validation."""

    def test_invalid_bbox_longitude(self):
        """Test that invalid longitude range raises ValueError."""
        config = AnalysisConfig(
            bbox=(-200, 32.45, -115.25, 32.55),  # Invalid min_lon
            date_from="2025-10-15"
        )

        with pytest.raises(ValueError, match="Invalid longitude range"):
            pipeline = HierarchicalAnalysisPipeline(config)

    def test_invalid_bbox_latitude(self):
        """Test that invalid latitude range raises ValueError."""
        config = AnalysisConfig(
            bbox=(-115.35, 100, -115.25, 32.55),  # Invalid min_lat
            date_from="2025-10-15"
        )

        with pytest.raises(ValueError, match="Invalid latitude range"):
            pipeline = HierarchicalAnalysisPipeline(config)

    def test_bbox_too_large(self):
        """Test that bbox > 0.1° x 0.1° raises ValueError."""
        config = AnalysisConfig(
            bbox=(-115.35, 32.45, -115.0, 32.8),  # 0.35° x 0.35°
            date_from="2025-10-15"
        )

        with pytest.raises(ValueError, match="BBox too large"):
            pipeline = HierarchicalAnalysisPipeline(config)

    def test_invalid_date_format(self):
        """Test that invalid date format raises ValueError."""
        config = AnalysisConfig(
            bbox=(-115.35, 32.45, -115.25, 32.55),
            date_from="10-15-2025"  # Invalid format
        )

        with pytest.raises(ValueError, match="Invalid date format"):
            pipeline = HierarchicalAnalysisPipeline(config)

    def test_valid_config_initialization(self, tmp_path):
        """Test that valid config initializes successfully."""
        config = AnalysisConfig(
            bbox=(-115.35, 32.45, -115.25, 32.55),
            date_from="2025-10-15",
            output_dir=str(tmp_path / "output")
        )

        pipeline = HierarchicalAnalysisPipeline(config)

        assert pipeline.config.bbox == (-115.35, 32.45, -115.25, 32.55)
        assert pipeline.config.date_from == "2025-10-15"
        assert pipeline.config.date_to == "2025-10-15"  # Auto-filled
        assert pipeline.output_dir.exists()


class TestHierarchicalPipelineMocked:
    """Test pipeline with mocked external dependencies."""

    @pytest.fixture
    def mock_hls_data(self):
        """Generate mock HLS data."""
        # Small image for testing: 100x100 pixels, 6 bands
        return np.random.rand(100, 100, 6).astype(np.float32)

    @pytest.fixture
    def mock_embeddings(self):
        """Generate mock Prithvi embeddings."""
        # 100x100 pixels, 256 dimensions
        return np.random.rand(100, 100, 256).astype(np.float32)

    @pytest.fixture
    def mock_segmentation(self):
        """Generate mock segmentation."""
        # 100x100 pixels with 10 regions
        seg = np.zeros((100, 100), dtype=np.int32)
        for i in range(10):
            y_start = (i // 5) * 50
            x_start = (i % 5) * 20
            seg[y_start:y_start+50, x_start:x_start+20] = i + 1
        return seg

    def test_pipeline_initialization(self, tmp_path):
        """Test that pipeline initializes with valid config."""
        config = AnalysisConfig(
            bbox=(-115.35, 32.45, -115.25, 32.55),
            date_from="2025-10-15",
            output_dir=str(tmp_path / "output")
        )

        pipeline = HierarchicalAnalysisPipeline(config)

        assert pipeline.config is not None
        assert pipeline.output_dir.exists()
        assert pipeline.step_times == {}

    def test_ndvi_calculation(self, tmp_path, mock_hls_data):
        """Test NDVI calculation from HLS data."""
        config = AnalysisConfig(
            bbox=(-115.35, 32.45, -115.25, 32.55),
            date_from="2025-10-15",
            output_dir=str(tmp_path / "output")
        )

        pipeline = HierarchicalAnalysisPipeline(config)
        ndvi = pipeline._calculate_ndvi(mock_hls_data)

        assert ndvi.shape == (100, 100)
        assert ndvi.dtype == np.float64
        assert -1.0 <= ndvi.min() <= ndvi.max() <= 1.0

        # Check that NDVI was saved
        ndvi_path = pipeline.output_dir / "ndvi.npy"
        assert ndvi_path.exists()

    def test_stress_analysis(self, tmp_path):
        """Test stress analysis for crop regions."""
        from src.classification.zero_shot_classifier import ClassificationResult

        config = AnalysisConfig(
            bbox=(-115.35, 32.45, -115.25, 32.55),
            date_from="2025-10-15",
            output_dir=str(tmp_path / "output")
        )

        pipeline = HierarchicalAnalysisPipeline(config)

        # Create mock classifications with different crop stress levels
        classifications = {
            1: ClassificationResult(
                class_id=3,  # Vigorous Crop
                class_name="Vigorous Crop (Cultivo Vigoroso)",
                confidence=0.85,
                mean_ndvi=0.70,  # Low stress
                std_ndvi=0.05,
                size_pixels=1000,
                area_hectares=10.0
            ),
            2: ClassificationResult(
                class_id=4,  # Stressed Crop
                class_name="Stressed Crop (Cultivo Estresado)",
                confidence=0.80,
                mean_ndvi=0.45,  # Medium stress
                std_ndvi=0.08,
                size_pixels=800,
                area_hectares=8.0
            ),
            3: ClassificationResult(
                class_id=4,  # Stressed Crop
                class_name="Stressed Crop (Cultivo Estresado)",
                confidence=0.75,
                mean_ndvi=0.30,  # High stress
                std_ndvi=0.10,
                size_pixels=600,
                area_hectares=6.0
            ),
            4: ClassificationResult(
                class_id=0,  # Water (not a crop)
                class_name="Water (Agua)",
                confidence=0.90,
                mean_ndvi=-0.20,
                std_ndvi=0.02,
                size_pixels=500,
                area_hectares=5.0
            ),
        }

        # Create mock NDVI and segmentation
        ndvi = np.random.rand(100, 100)
        segmentation = np.zeros((100, 100), dtype=np.int32)

        # Analyze stress
        stress_results = pipeline._analyze_stress(classifications, ndvi, segmentation)

        # Verify results
        assert stress_results['low']['count'] == 1  # Region 1
        assert stress_results['medium']['count'] == 1  # Region 2
        assert stress_results['high']['count'] == 1  # Region 3
        assert stress_results['low']['area_ha'] == 10.0
        assert stress_results['medium']['area_ha'] == 8.0
        assert stress_results['high']['area_ha'] == 6.0

    def test_json_output_generation(self, tmp_path):
        """Test JSON output generation."""
        from src.classification.zero_shot_classifier import ClassificationResult

        config = AnalysisConfig(
            bbox=(-115.35, 32.45, -115.25, 32.55),
            date_from="2025-10-15",
            output_dir=str(tmp_path / "output")
        )

        pipeline = HierarchicalAnalysisPipeline(config)

        # Create mock data
        classifications = {
            1: ClassificationResult(
                class_id=3,
                class_name="Vigorous Crop (Cultivo Vigoroso)",
                confidence=0.85,
                mean_ndvi=0.70,
                std_ndvi=0.05,
                size_pixels=1000,
                area_hectares=10.0
            ),
        }

        stress_results = {
            'low': {'count': 1, 'area_ha': 10.0, 'regions': []},
            'medium': {'count': 0, 'area_ha': 0.0, 'regions': []},
            'high': {'count': 0, 'area_ha': 0.0, 'regions': []},
        }

        # Generate JSON
        json_path = pipeline.output_dir / "test_results.json"
        pipeline._save_json(classifications, stress_results, json_path)

        # Verify JSON was created
        assert json_path.exists()

        # Load and verify content
        import json
        with open(json_path) as f:
            data = json.load(f)

        assert 'metadata' in data
        assert 'segmentation' in data
        assert 'classification' in data
        assert 'stress_analysis' in data
        assert 'summary' in data
        assert len(data['classification']) == 1


class TestHierarchicalPipelineIntegration:
    """Integration tests with real components (but small data)."""

    def test_pipeline_components_integration(self, tmp_path, monkeypatch):
        """
        Test that pipeline components integrate correctly.

        Uses mocks for external dependencies (Sentinel Hub, Prithvi model)
        but tests real integration of our components.
        """
        # Mock download_sentinel2_bands
        def mock_download(bbox_coords, config, bands, date_from, date_to, resolution):
            # Return small mock data
            hls_data = np.random.rand(6, 100, 100).astype(np.float32)
            metadata = {
                'bbox': bbox_coords,
                'date': date_from,
                'resolution': resolution
            }
            bands_dict = {
                'B02': hls_data[0],
                'B03': hls_data[1],
                'B04': hls_data[2],
                'B8A': hls_data[3],
                'B11': hls_data[4],
                'B12': hls_data[5]
            }
            return {'bands': bands_dict, 'metadata': metadata}

        monkeypatch.setattr(
            "src.pipeline.hierarchical_analysis.download_sentinel2_bands",
            mock_download
        )

        # Mock Prithvi model
        def mock_load_model():
            class MockModel:
                pass
            return MockModel()

        def mock_extract_embeddings(model, hls_data):
            # Return mock embeddings
            h, w, c = hls_data.shape
            return np.random.rand(h, w, 256).astype(np.float32)

        monkeypatch.setattr(
            "src.pipeline.hierarchical_analysis.load_prithvi_model",
            mock_load_model
        )
        monkeypatch.setattr(
            "src.pipeline.hierarchical_analysis.extract_embeddings_from_hls",
            mock_extract_embeddings
        )

        # Set environment variables
        monkeypatch.setenv('SH_CLIENT_ID', 'test_id')
        monkeypatch.setenv('SH_CLIENT_SECRET', 'test_secret')

        # Create config
        config = AnalysisConfig(
            bbox=(-115.35, 32.45, -115.30, 32.50),  # Small bbox
            date_from="2025-10-15",
            output_dir=str(tmp_path / "output"),
            export_formats=["json"]  # Only JSON for speed
        )

        # Run pipeline
        pipeline = HierarchicalAnalysisPipeline(config)
        result = pipeline.run()

        # Verify result structure
        assert isinstance(result, AnalysisResult)
        assert result.metadata is not None
        assert result.segmentation is not None
        assert result.classification is not None
        assert result.stress_analysis is not None
        assert result.summary is not None
        assert result.output_files is not None
        assert result.processing_time is not None

        # Verify output files
        assert 'json' in result.output_files
        json_path = Path(result.output_files['json'])
        assert json_path.exists()

        # Verify processing times
        assert 'download' in result.processing_time
        assert 'embeddings' in result.processing_time
        assert 'segmentation' in result.processing_time
        assert 'ndvi' in result.processing_time
        assert 'classification' in result.processing_time
        assert 'stress' in result.processing_time
        assert 'output' in result.processing_time
        assert 'total' in result.processing_time


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
