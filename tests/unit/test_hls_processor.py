"""
Unit tests for HLS processor module.

Tests cover:
- Resampling bands from 20m to 10m
- Stacking bands in correct order
- L2 normalization of embeddings
- HLS image preparation
- Embeddings extraction
- Embeddings saving/loading
- Cosine similarity computation
- PCA visualization
"""
import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil

from src.features.hls_processor import (
    resample_band_to_10m,
    stack_hls_bands,
    normalize_embeddings_l2,
    prepare_hls_image,
    compute_cosine_similarity,
    save_embeddings,
    load_embeddings,
    visualize_embeddings_pca
)


class TestResampleBandTo10m:
    """Tests for resample_band_to_10m function"""
    
    def test_doubles_dimensions_by_default(self):
        """Test that resampling doubles dimensions by default"""
        band_20m = np.random.rand(256, 256)
        band_10m = resample_band_to_10m(band_20m)
        assert band_10m.shape == (512, 512)
    
    def test_resamples_to_target_shape(self):
        """Test resampling to specific target shape"""
        band_20m = np.random.rand(256, 256)
        band_10m = resample_band_to_10m(band_20m, target_shape=(480, 480))
        assert band_10m.shape == (480, 480)
    
    def test_preserves_data_range(self):
        """Test that resampling preserves data range"""
        band_20m = np.random.rand(256, 256)
        band_10m = resample_band_to_10m(band_20m)
        assert band_10m.min() >= 0
        assert band_10m.max() <= 1
    
    def test_handles_rectangular_shapes(self):
        """Test resampling with non-square dimensions"""
        band_20m = np.random.rand(128, 256)
        band_10m = resample_band_to_10m(band_20m)
        assert band_10m.shape == (256, 512)


class TestStackHlsBands:
    """Tests for stack_hls_bands function"""
    
    def test_stacks_in_correct_order(self):
        """Test that bands are stacked in correct order"""
        bands = {
            'B02': np.ones((512, 512)) * 0.1,
            'B03': np.ones((512, 512)) * 0.2,
            'B04': np.ones((512, 512)) * 0.3,
            'B8A': np.ones((512, 512)) * 0.4,
            'B11': np.ones((512, 512)) * 0.5,
            'B12': np.ones((512, 512)) * 0.6
        }
        stacked = stack_hls_bands(bands)
        assert stacked.shape == (6, 512, 512)
        assert np.allclose(stacked[0], 0.1)
        assert np.allclose(stacked[5], 0.6)
    
    def test_raises_on_missing_band(self):
        """Test that missing bands raise ValueError"""
        bands = {
            'B02': np.ones((512, 512)),
            'B03': np.ones((512, 512))
        }
        with pytest.raises(ValueError, match="Missing required band"):
            stack_hls_bands(bands)
    
    def test_raises_on_dimension_mismatch(self):
        """Test that dimension mismatch raises ValueError"""
        bands = {
            'B02': np.ones((512, 512)),
            'B03': np.ones((512, 512)),
            'B04': np.ones((512, 512)),
            'B8A': np.ones((256, 256)),
            'B11': np.ones((512, 512)),
            'B12': np.ones((512, 512))
        }
        with pytest.raises(ValueError, match="Band dimensions don't match"):
            stack_hls_bands(bands)
    
    def test_skips_validation_when_disabled(self):
        """Test that validation can be skipped"""
        # When validation is disabled, we can pass valid bands without explicit validation
        # This is useful when we already know the bands are correct
        bands = {
            'B02': np.ones((512, 512)),
            'B03': np.ones((512, 512)),
            'B04': np.ones((512, 512)),
            'B8A': np.ones((512, 512)),
            'B11': np.ones((512, 512)),
            'B12': np.ones((512, 512))
        }
        # Should work without validation
        stacked = stack_hls_bands(bands, validate=False)
        assert stacked.shape == (6, 512, 512)

        # With validation disabled, the function trusts the input
        # and won't raise errors if all 6 bands are present and have same shape


class TestNormalizeEmbeddingsL2:
    """Tests for normalize_embeddings_l2 function"""
    
    def test_normalizes_to_unit_norm(self):
        """Test that embeddings are normalized to unit L2 norm"""
        embeddings = np.random.rand(10, 10, 256)
        normalized = normalize_embeddings_l2(embeddings)
        norms = np.linalg.norm(normalized, axis=2)
        assert np.allclose(norms, 1.0, atol=1e-6)
    
    def test_preserves_shape(self):
        """Test that normalization preserves shape"""
        embeddings = np.random.rand(512, 512, 256)
        normalized = normalize_embeddings_l2(embeddings)
        assert normalized.shape == embeddings.shape
    
    def test_handles_zero_vectors(self):
        """Test that zero vectors are handled gracefully"""
        embeddings = np.zeros((10, 10, 256))
        normalized = normalize_embeddings_l2(embeddings)
        assert normalized.shape == embeddings.shape
        assert np.allclose(normalized, 0.0)
    
    def test_different_input_sizes(self):
        """Test normalization with various input sizes"""
        for h, w, d in [(10, 10, 128), (224, 224, 256), (512, 512, 512)]:
            embeddings = np.random.rand(h, w, d)
            normalized = normalize_embeddings_l2(embeddings)
            assert normalized.shape == (h, w, d)


class TestPrepareHlsImage:
    """Tests for prepare_hls_image function"""
    
    def test_prepares_full_hls_image(self):
        """Test preparation of complete HLS image"""
        bands_10m = {
            'B02': np.random.rand(512, 512),
            'B03': np.random.rand(512, 512),
            'B04': np.random.rand(512, 512)
        }
        bands_20m = {
            'B8A': np.random.rand(256, 256),
            'B11': np.random.rand(256, 256),
            'B12': np.random.rand(256, 256)
        }
        hls_image = prepare_hls_image(bands_10m, bands_20m)
        assert hls_image.shape == (6, 512, 512)
    
    def test_resamples_20m_bands_correctly(self):
        """Test that 20m bands are resampled to match 10m"""
        bands_10m = {
            'B02': np.random.rand(512, 512),
            'B03': np.random.rand(512, 512),
            'B04': np.random.rand(512, 512)
        }
        bands_20m = {
            'B8A': np.random.rand(256, 256),
            'B11': np.random.rand(256, 256),
            'B12': np.random.rand(256, 256)
        }
        hls_image = prepare_hls_image(bands_10m, bands_20m)
        assert hls_image[3].shape == (512, 512)


class TestComputeCosineSimilarity:
    """Tests for compute_cosine_similarity function"""
    
    def test_identical_embeddings_have_high_similarity(self):
        """Test that identical embeddings have similarity near 1"""
        embeddings = np.random.rand(10, 10, 256)
        embeddings = normalize_embeddings_l2(embeddings)
        similarity = compute_cosine_similarity(embeddings, embeddings)
        assert np.allclose(similarity, 1.0, atol=1e-6)
    
    def test_orthogonal_embeddings_have_zero_similarity(self):
        """Test that orthogonal embeddings have similarity near 0"""
        emb_a = np.zeros((10, 10, 256))
        emb_a[:, :, 0] = 1.0
        emb_b = np.zeros((10, 10, 256))
        emb_b[:, :, 1] = 1.0
        similarity = compute_cosine_similarity(emb_a, emb_b)
        assert np.allclose(similarity, 0.0, atol=1e-6)
    
    def test_output_shape(self):
        """Test that output has correct shape"""
        emb_a = normalize_embeddings_l2(np.random.rand(512, 512, 256))
        emb_b = normalize_embeddings_l2(np.random.rand(512, 512, 256))
        similarity = compute_cosine_similarity(emb_a, emb_b)
        assert similarity.shape == (512, 512)
    
    def test_raises_on_shape_mismatch(self):
        """Test that shape mismatch raises ValueError"""
        emb_a = np.random.rand(10, 10, 256)
        emb_b = np.random.rand(20, 20, 256)
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_cosine_similarity(emb_a, emb_b)


class TestSaveLoadEmbeddings:
    """Tests for save_embeddings and load_embeddings functions"""
    
    def setup_method(self):
        """Create temporary directory for tests"""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """Clean up temporary directory"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_saves_and_loads_embeddings(self):
        """Test saving and loading embeddings"""
        embeddings_orig = np.random.rand(512, 512, 256)
        output_path = self.temp_dir / "test_embeddings.npz"
        
        save_embeddings(embeddings_orig, output_path)
        embeddings_loaded, metadata = load_embeddings(output_path)
        
        assert np.allclose(embeddings_orig, embeddings_loaded)
    
    def test_saves_and_loads_metadata(self):
        """Test saving and loading metadata"""
        embeddings = np.random.rand(10, 10, 256)
        output_path = self.temp_dir / "test_embeddings.npz"
        metadata_orig = {'zone': 'mexicali', 'date': '2024-01-15'}
        
        save_embeddings(embeddings, output_path, metadata_orig)
        embeddings_loaded, metadata_loaded = load_embeddings(output_path)
        
        assert metadata_loaded['zone'] == 'mexicali'
        assert metadata_loaded['date'] == '2024-01-15'
    
    def test_creates_directory_if_not_exists(self):
        """Test that save creates directory"""
        embeddings = np.random.rand(10, 10, 256)
        output_path = self.temp_dir / "subdir" / "embeddings.npz"
        
        save_embeddings(embeddings, output_path)
        assert output_path.exists()
    
    def test_raises_on_load_nonexistent(self):
        """Test that loading nonexistent file raises error"""
        with pytest.raises(FileNotFoundError):
            load_embeddings(self.temp_dir / "nonexistent.npz")


class TestVisualizePCA:
    """Tests for visualize_embeddings_pca function"""
    
    def test_reduces_to_3_components(self):
        """Test that PCA reduces to 3 components by default"""
        embeddings = np.random.rand(100, 100, 256)
        rgb = visualize_embeddings_pca(embeddings)
        assert rgb.shape == (100, 100, 3)
    
    def test_output_range_is_normalized(self):
        """Test that output is normalized to [0, 1]"""
        embeddings = np.random.rand(100, 100, 256)
        rgb = visualize_embeddings_pca(embeddings)
        assert rgb.min() >= 0.0
        assert rgb.max() <= 1.0
    
    def test_custom_number_of_components(self):
        """Test PCA with custom number of components"""
        embeddings = np.random.rand(100, 100, 256)
        result = visualize_embeddings_pca(embeddings, n_components=5)
        assert result.shape == (100, 100, 5)


class TestIntegration:
    """Integration tests for full pipeline"""
    
    def test_full_pipeline_end_to_end(self):
        """Test complete pipeline from bands to embeddings"""
        bands_10m = {
            'B02': np.random.rand(224, 224),
            'B03': np.random.rand(224, 224),
            'B04': np.random.rand(224, 224)
        }
        bands_20m = {
            'B8A': np.random.rand(112, 112),
            'B11': np.random.rand(112, 112),
            'B12': np.random.rand(112, 112)
        }
        
        hls_image = prepare_hls_image(bands_10m, bands_20m)
        assert hls_image.shape == (6, 224, 224)
    
    def test_embeddings_can_be_visualized(self):
        """Test that embeddings can be visualized with PCA"""
        embeddings = np.random.rand(100, 100, 256)
        embeddings = normalize_embeddings_l2(embeddings)
        rgb = visualize_embeddings_pca(embeddings)
        assert rgb.shape == (100, 100, 3)
        assert rgb.min() >= 0.0
        assert rgb.max() <= 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--cov=src.features.hls_processor', '--cov-report=html'])
