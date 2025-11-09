"""
Unit tests for Classic Region Growing algorithm.

Tests cover normal operation, edge cases, error handling, and algorithm correctness.
"""
import pytest
import numpy as np
from src.algorithms.classic_region_growing import ClassicRegionGrowing


class TestClassicRegionGrowing:
    """Test suite for ClassicRegionGrowing class."""

    def test_initialization_default_parameters(self):
        """Test that algorithm initializes with default parameters."""
        algorithm = ClassicRegionGrowing()

        assert algorithm.threshold == 0.1
        assert algorithm.min_region_size == 50
        assert algorithm.cloud_mask_value == -999.0

    def test_initialization_custom_parameters(self):
        """Test that algorithm initializes with custom parameters."""
        algorithm = ClassicRegionGrowing(
            threshold=0.05,
            min_region_size=100,
            cloud_mask_value=-9999.0
        )

        assert algorithm.threshold == 0.05
        assert algorithm.min_region_size == 100
        assert algorithm.cloud_mask_value == -9999.0

    def test_initialization_invalid_threshold(self):
        """Test that invalid threshold raises ValueError."""
        with pytest.raises(ValueError, match="Threshold must be positive"):
            ClassicRegionGrowing(threshold=0)

        with pytest.raises(ValueError, match="Threshold must be positive"):
            ClassicRegionGrowing(threshold=-0.1)

    def test_initialization_invalid_min_region_size(self):
        """Test that invalid min_region_size raises ValueError."""
        with pytest.raises(ValueError, match="Minimum region size must be at least 1"):
            ClassicRegionGrowing(min_region_size=0)

    def test_segment_single_region_homogeneous_image(self):
        """Test that a completely homogeneous image produces one region."""
        # Arrange
        image = np.ones((100, 100)) * 0.5
        algorithm = ClassicRegionGrowing(threshold=0.1, min_region_size=10)

        # Act
        labeled, num_regions, regions_info = algorithm.segment(image)

        # Assert
        assert num_regions == 1
        assert regions_info[0]['id'] == 1
        assert regions_info[0]['size'] == 10000  # 100x100
        assert abs(regions_info[0]['mean_ndvi'] - 0.5) < 0.01
        assert regions_info[0]['std_ndvi'] < 0.01  # Should be very low variance
        assert np.all(labeled > 0)  # All pixels should be labeled

    def test_segment_two_regions_distinct_values(self):
        """Test segmentation of image with two distinct regions."""
        # Arrange: Left half = 0.3, Right half = 0.7
        image = np.zeros((100, 100))
        image[:, :50] = 0.3
        image[:, 50:] = 0.7
        algorithm = ClassicRegionGrowing(threshold=0.1, min_region_size=10)

        # Act
        labeled, num_regions, regions_info = algorithm.segment(image)

        # Assert
        assert num_regions == 2
        # Sort regions by mean NDVI for consistent comparison
        sorted_regions = sorted(regions_info, key=lambda r: r['mean_ndvi'])
        assert sorted_regions[0]['size'] == 5000
        assert sorted_regions[1]['size'] == 5000
        assert abs(sorted_regions[0]['mean_ndvi'] - 0.3) < 0.01
        assert abs(sorted_regions[1]['mean_ndvi'] - 0.7) < 0.01

    def test_segment_multiple_regions_gradient(self):
        """Test segmentation of image with gradient produces multiple regions."""
        # Arrange: Horizontal gradient from 0.2 to 0.8
        image = np.zeros((100, 100))
        for i in range(100):
            image[:, i] = 0.2 + (0.6 * i / 99)

        algorithm = ClassicRegionGrowing(threshold=0.1, min_region_size=50)

        # Act
        labeled, num_regions, regions_info = algorithm.segment(image)

        # Assert
        assert num_regions >= 3  # Gradient should create multiple regions
        assert num_regions <= 10  # But not too many with threshold=0.1
        # Check that NDVI increases across regions
        sorted_regions = sorted(regions_info, key=lambda r: r['mean_ndvi'])
        for i in range(len(sorted_regions) - 1):
            assert sorted_regions[i]['mean_ndvi'] < sorted_regions[i+1]['mean_ndvi']

    def test_segment_filter_small_regions(self):
        """Test that regions smaller than min_size are filtered out."""
        # Arrange: Mostly 0.5, with small patches of 0.8
        image = np.ones((100, 100)) * 0.5
        image[10:15, 10:15] = 0.8  # 25 pixels
        image[80:90, 80:90] = 0.8  # 100 pixels
        algorithm = ClassicRegionGrowing(threshold=0.1, min_region_size=50)

        # Act
        labeled, num_regions, regions_info = algorithm.segment(image)

        # Assert
        assert num_regions >= 1  # At least one region (may merge if seeds connect them)
        # All regions should meet minimum size
        for region in regions_info:
            assert region['size'] >= 50
        # Total labeled pixels should be reasonable
        total_labeled = sum(r['size'] for r in regions_info)
        assert total_labeled > 0

    def test_segment_cloud_mask_handling(self):
        """Test that masked pixels (clouds) are ignored."""
        # Arrange
        image = np.ones((100, 100)) * 0.5
        image[40:60, 40:60] = -999.0  # Cloud mask value
        algorithm = ClassicRegionGrowing(threshold=0.1, min_region_size=10)

        # Act
        labeled, num_regions, regions_info = algorithm.segment(image)

        # Assert
        # Should create regions around the masked area
        assert num_regions >= 1
        # Masked pixels should not be in any region
        assert np.all(labeled[40:60, 40:60] == 0)
        # Check that total labeled pixels + masked pixels ≈ image size
        total_labeled = np.sum(labeled > 0)
        masked_pixels = 20 * 20  # 400 pixels
        assert total_labeled + masked_pixels <= 10000

    def test_segment_with_custom_seeds(self):
        """Test segmentation with custom seed points."""
        # Arrange
        image = np.ones((100, 100)) * 0.5
        custom_seeds = [(10, 10), (50, 50), (90, 90)]
        algorithm = ClassicRegionGrowing(threshold=0.1, min_region_size=10)

        # Act
        labeled, num_regions, regions_info = algorithm.segment(image, seeds=custom_seeds)

        # Assert
        # With homogeneous image and strict threshold, should create one region
        assert num_regions == 1
        assert regions_info[0]['size'] == 10000

    def test_segment_invalid_image_shape(self):
        """Test that non-2D image raises ValueError."""
        # Arrange
        image_3d = np.ones((100, 100, 3))
        algorithm = ClassicRegionGrowing()

        # Act & Assert
        with pytest.raises(ValueError, match="Image must be 2D"):
            algorithm.segment(image_3d)

    def test_generate_seeds_grid_spacing(self):
        """Test that grid seed generation creates evenly spaced seeds."""
        # Arrange
        image = np.ones((100, 100)) * 0.5
        algorithm = ClassicRegionGrowing()

        # Act
        seeds = algorithm.generate_seeds_grid(image, grid_size=20)

        # Assert
        assert len(seeds) == 25  # 5x5 grid for 100x100 image with spacing 20
        # Check first few seeds are at expected positions
        assert (10, 10) in seeds
        assert (10, 30) in seeds
        assert (30, 10) in seeds

    def test_generate_seeds_grid_skips_masked_pixels(self):
        """Test that grid seed generation skips masked pixels."""
        # Arrange
        image = np.ones((100, 100)) * 0.5
        image[10, 10] = -999.0  # Mask one seed location
        algorithm = ClassicRegionGrowing()

        # Act
        seeds = algorithm.generate_seeds_grid(image, grid_size=20)

        # Assert
        assert (10, 10) not in seeds  # Masked pixel should be skipped
        assert len(seeds) == 24  # One less than expected

    def test_classify_by_stress_high_stress(self):
        """Test classification of regions with high stress (NDVI < 0.3)."""
        # Arrange
        regions_info = [
            {'id': 1, 'mean_ndvi': 0.15, 'size': 100},
            {'id': 2, 'mean_ndvi': 0.25, 'size': 200}
        ]
        algorithm = ClassicRegionGrowing()

        # Act
        classified = algorithm.classify_by_stress(regions_info)

        # Assert
        assert len(classified['high_stress']) == 2
        assert len(classified['medium_stress']) == 0
        assert len(classified['low_stress']) == 0
        assert regions_info[0]['stress_level'] == 'high'
        assert regions_info[1]['stress_level'] == 'high'

    def test_classify_by_stress_medium_stress(self):
        """Test classification of regions with medium stress (0.3 ≤ NDVI < 0.5)."""
        # Arrange
        regions_info = [
            {'id': 1, 'mean_ndvi': 0.35, 'size': 100},
            {'id': 2, 'mean_ndvi': 0.45, 'size': 200}
        ]
        algorithm = ClassicRegionGrowing()

        # Act
        classified = algorithm.classify_by_stress(regions_info)

        # Assert
        assert len(classified['high_stress']) == 0
        assert len(classified['medium_stress']) == 2
        assert len(classified['low_stress']) == 0

    def test_classify_by_stress_low_stress(self):
        """Test classification of regions with low stress (NDVI ≥ 0.5)."""
        # Arrange
        regions_info = [
            {'id': 1, 'mean_ndvi': 0.55, 'size': 100},
            {'id': 2, 'mean_ndvi': 0.75, 'size': 200}
        ]
        algorithm = ClassicRegionGrowing()

        # Act
        classified = algorithm.classify_by_stress(regions_info)

        # Assert
        assert len(classified['high_stress']) == 0
        assert len(classified['medium_stress']) == 0
        assert len(classified['low_stress']) == 2

    def test_classify_by_stress_mixed_levels(self):
        """Test classification with mixed stress levels."""
        # Arrange
        regions_info = [
            {'id': 1, 'mean_ndvi': 0.2, 'size': 100},   # High stress
            {'id': 2, 'mean_ndvi': 0.4, 'size': 200},   # Medium stress
            {'id': 3, 'mean_ndvi': 0.6, 'size': 300}    # Low stress
        ]
        algorithm = ClassicRegionGrowing()

        # Act
        classified = algorithm.classify_by_stress(regions_info)

        # Assert
        assert len(classified['high_stress']) == 1
        assert len(classified['medium_stress']) == 1
        assert len(classified['low_stress']) == 1
        assert classified['high_stress'][0]['id'] == 1
        assert classified['medium_stress'][0]['id'] == 2
        assert classified['low_stress'][0]['id'] == 3

    def test_segment_with_nan_values(self):
        """Test that NaN values are treated as masked pixels."""
        # Arrange
        image = np.ones((100, 100)) * 0.5
        image[40:60, 40:60] = np.nan
        algorithm = ClassicRegionGrowing(threshold=0.1, min_region_size=10)

        # Act
        labeled, num_regions, regions_info = algorithm.segment(image)

        # Assert
        # NaN pixels should not be in any region
        assert np.all(labeled[40:60, 40:60] == 0)
        assert num_regions >= 1

    def test_segment_with_inf_values(self):
        """Test that Inf values are treated as masked pixels."""
        # Arrange
        image = np.ones((100, 100)) * 0.5
        image[40:60, 40:60] = np.inf
        algorithm = ClassicRegionGrowing(threshold=0.1, min_region_size=10)

        # Act
        labeled, num_regions, regions_info = algorithm.segment(image)

        # Assert
        # Inf pixels should not be in any region
        assert np.all(labeled[40:60, 40:60] == 0)
        assert num_regions >= 1

    def test_segment_empty_image(self):
        """Test segmentation of image with all masked pixels."""
        # Arrange
        image = np.ones((100, 100)) * -999.0  # All masked
        algorithm = ClassicRegionGrowing(threshold=0.1, min_region_size=10)

        # Act
        labeled, num_regions, regions_info = algorithm.segment(image)

        # Assert
        assert num_regions == 0
        assert len(regions_info) == 0
        assert np.all(labeled == 0)

    def test_segment_algorithm_determinism(self):
        """Test that algorithm produces same results on identical input."""
        # Arrange
        np.random.seed(42)
        image = np.random.rand(100, 100)
        algorithm = ClassicRegionGrowing(threshold=0.1, min_region_size=50)

        # Act
        labeled1, num_regions1, regions_info1 = algorithm.segment(image)
        labeled2, num_regions2, regions_info2 = algorithm.segment(image)

        # Assert
        assert num_regions1 == num_regions2
        assert len(regions_info1) == len(regions_info2)
        assert np.array_equal(labeled1, labeled2)

    def test_segment_performance_large_image(self):
        """Test that algorithm can handle large images efficiently."""
        # Arrange - create more structured image with clear regions
        image = np.zeros((512, 512))
        image[:256, :] = 0.3  # Top half
        image[256:, :] = 0.7  # Bottom half
        algorithm = ClassicRegionGrowing(threshold=0.1, min_region_size=100)

        # Act & Assert
        # Should complete in reasonable time (< 5 seconds)
        import time
        start = time.time()
        labeled, num_regions, regions_info = algorithm.segment(image)
        elapsed = time.time() - start

        assert elapsed < 5.0  # Should be fast on modern hardware
        assert num_regions >= 1  # Should find at least one region
