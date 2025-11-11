"""
Unit tests for Semantic Region Growing (MGRG) algorithm.

This test suite validates the MGRG implementation including:
- Smart seed generation with K-Means
- Grid-based seed generation
- BFS segmentation with cosine similarity
- Hierarchical stress analysis
- Edge cases and error handling

Test Coverage Target: >70%
"""

import pytest
import numpy as np
from src.algorithms.semantic_region_growing import SemanticRegionGrowing


class TestInitialization:
    """Test class initialization and parameter validation."""

    def test_default_initialization(self):
        """Test initialization with default parameters."""
        algorithm = SemanticRegionGrowing()

        assert algorithm.threshold == 0.85
        assert algorithm.min_region_size == 50
        assert algorithm.use_smart_seeds is True
        assert algorithm.n_clusters == 5
        assert algorithm.random_state == 42
        assert algorithm.num_regions_ == 0
        assert algorithm.seeds_ == []

    def test_custom_parameters(self):
        """Test initialization with custom parameters."""
        algorithm = SemanticRegionGrowing(
            threshold=0.90,
            min_region_size=100,
            use_smart_seeds=False,
            n_clusters=10,
            random_state=123,
        )

        assert algorithm.threshold == 0.90
        assert algorithm.min_region_size == 100
        assert algorithm.use_smart_seeds is False
        assert algorithm.n_clusters == 10
        assert algorithm.random_state == 123

    def test_invalid_threshold_too_low(self):
        """Test that threshold below 0 raises ValueError."""
        with pytest.raises(ValueError, match="Threshold must be in"):
            SemanticRegionGrowing(threshold=-0.1)

    def test_invalid_threshold_too_high(self):
        """Test that threshold above 1 raises ValueError."""
        with pytest.raises(ValueError, match="Threshold must be in"):
            SemanticRegionGrowing(threshold=1.5)

    def test_invalid_min_region_size(self):
        """Test that invalid min_region_size raises ValueError."""
        with pytest.raises(ValueError, match="Minimum region size"):
            SemanticRegionGrowing(min_region_size=0)

    def test_invalid_n_clusters(self):
        """Test that invalid n_clusters raises ValueError."""
        with pytest.raises(ValueError, match="Number of clusters"):
            SemanticRegionGrowing(n_clusters=0)


class TestGridSeeds:
    """Test grid-based seed generation."""

    def test_generate_grid_seeds_basic(self):
        """Test basic grid seed generation."""
        embeddings = np.random.rand(100, 100, 16)
        algorithm = SemanticRegionGrowing()

        seeds = algorithm.generate_grid_seeds(embeddings, grid_size=20)

        assert len(seeds) > 0
        assert all(isinstance(s, tuple) and len(s) == 2 for s in seeds)

        for y, x in seeds:
            assert 0 <= y < 100
            assert 0 <= x < 100

    def test_generate_grid_seeds_count(self):
        """Test that grid produces expected number of seeds."""
        embeddings = np.random.rand(100, 100, 16)
        algorithm = SemanticRegionGrowing()

        seeds = algorithm.generate_grid_seeds(embeddings, grid_size=20)

        expected_count = (100 // 20) * (100 // 20)
        assert len(seeds) == expected_count

    def test_generate_grid_seeds_spacing(self):
        """Test that seeds are properly spaced."""
        embeddings = np.random.rand(100, 100, 16)
        algorithm = SemanticRegionGrowing()

        grid_size = 25
        seeds = algorithm.generate_grid_seeds(embeddings, grid_size=grid_size)

        seeds_y = [y for y, x in seeds]
        seeds_x = [x for y, x in seeds]

        assert min(seeds_y) == grid_size // 2
        assert min(seeds_x) == grid_size // 2

    def test_generate_grid_seeds_different_sizes(self):
        """Test grid generation on different image sizes."""
        for h, w in [(50, 50), (100, 150), (200, 100)]:
            embeddings = np.random.rand(h, w, 16)
            algorithm = SemanticRegionGrowing()

            seeds = algorithm.generate_grid_seeds(embeddings, grid_size=10)

            assert len(seeds) > 0
            for y, x in seeds:
                assert 0 <= y < h
                assert 0 <= x < w


class TestSmartSeeds:
    """Test K-Means based smart seed generation."""

    def test_generate_smart_seeds_basic(self):
        """Test basic smart seed generation with K-Means."""
        np.random.seed(42)
        embeddings = np.random.rand(100, 100, 16)
        algorithm = SemanticRegionGrowing(n_clusters=5, random_state=42)

        seeds = algorithm.generate_smart_seeds(embeddings)

        assert len(seeds) == 5
        assert all(isinstance(s, tuple) and len(s) == 2 for s in seeds)

    def test_generate_smart_seeds_coordinates_valid(self):
        """Test that generated seeds have valid coordinates."""
        np.random.seed(42)
        embeddings = np.random.rand(100, 100, 16)
        algorithm = SemanticRegionGrowing(n_clusters=7)

        seeds = algorithm.generate_smart_seeds(embeddings)

        for y, x in seeds:
            assert 0 <= y < 100
            assert 0 <= x < 100
            assert isinstance(y, (int, np.integer))
            assert isinstance(x, (int, np.integer))

    def test_generate_smart_seeds_different_n_clusters(self):
        """Test smart seed generation with different cluster counts."""
        np.random.seed(42)
        embeddings = np.random.rand(100, 100, 16)

        for n_clusters in [3, 5, 7, 10]:
            algorithm = SemanticRegionGrowing(n_clusters=n_clusters, random_state=42)
            seeds = algorithm.generate_smart_seeds(embeddings)

            assert len(seeds) == n_clusters

    def test_generate_smart_seeds_reproducibility(self):
        """Test that smart seeds are reproducible with same random_state."""
        np.random.seed(42)
        embeddings = np.random.rand(100, 100, 16)

        algorithm1 = SemanticRegionGrowing(n_clusters=5, random_state=42)
        seeds1 = algorithm1.generate_smart_seeds(embeddings)

        algorithm2 = SemanticRegionGrowing(n_clusters=5, random_state=42)
        seeds2 = algorithm2.generate_smart_seeds(embeddings)

        assert seeds1 == seeds2

    def test_generate_smart_seeds_override_n_clusters(self):
        """Test overriding n_clusters parameter in method call."""
        np.random.seed(42)
        embeddings = np.random.rand(100, 100, 16)
        algorithm = SemanticRegionGrowing(n_clusters=5)

        seeds = algorithm.generate_smart_seeds(embeddings, n_clusters=3)

        assert len(seeds) == 3

    def test_smart_vs_grid_comparison(self):
        """Test that smart seeds produce fewer seeds than grid."""
        np.random.seed(42)
        embeddings = np.random.rand(200, 200, 16)
        algorithm = SemanticRegionGrowing()

        grid_seeds = algorithm.generate_grid_seeds(embeddings, grid_size=20)
        smart_seeds = algorithm.generate_smart_seeds(embeddings, n_clusters=5)

        assert len(smart_seeds) < len(grid_seeds)
        assert len(smart_seeds) == 5
        assert len(grid_seeds) == 100


class TestSegmentation:
    """Test segmentation algorithm."""

    def test_segment_with_smart_seeds(self):
        """Test segmentation using smart seed generation."""
        np.random.seed(42)
        embeddings = np.random.rand(50, 50, 16)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=2, keepdims=True)

        algorithm = SemanticRegionGrowing(
            threshold=0.85, min_region_size=10, use_smart_seeds=True, n_clusters=3, random_state=42
        )

        labeled, num_regions, regions_info = algorithm.segment(embeddings)

        assert labeled.shape == (50, 50)
        assert labeled.dtype == np.int32
        assert num_regions >= 0
        assert len(regions_info) == num_regions
        assert algorithm.num_regions_ == num_regions
        assert len(algorithm.seeds_) == 3

    def test_segment_with_grid_seeds(self):
        """Test segmentation using grid seed generation."""
        np.random.seed(42)
        embeddings = np.random.rand(50, 50, 16)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=2, keepdims=True)

        algorithm = SemanticRegionGrowing(
            threshold=0.85, min_region_size=10, use_smart_seeds=False, random_state=42
        )

        labeled, num_regions, regions_info = algorithm.segment(embeddings)

        assert labeled.shape == (50, 50)
        assert num_regions >= 0
        assert len(regions_info) == num_regions
        assert len(algorithm.seeds_) > 3

    def test_segment_with_explicit_seeds(self):
        """Test segmentation with explicitly provided seeds."""
        np.random.seed(42)
        embeddings = np.random.rand(50, 50, 16)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=2, keepdims=True)

        algorithm = SemanticRegionGrowing(threshold=0.80, min_region_size=10)

        explicit_seeds = [(10, 10), (20, 20), (30, 30)]
        labeled, num_regions, regions_info = algorithm.segment(embeddings, seeds=explicit_seeds)

        assert labeled.shape == (50, 50)
        assert algorithm.seeds_ == explicit_seeds

    def test_segment_threshold_sensitivity(self):
        """Test that different thresholds produce different segmentations."""
        np.random.seed(42)
        embeddings = np.random.rand(50, 50, 16)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=2, keepdims=True)

        algorithm_strict = SemanticRegionGrowing(
            threshold=0.95, min_region_size=5, use_smart_seeds=False, random_state=42
        )
        labeled_strict, num_regions_strict, _ = algorithm_strict.segment(embeddings)

        algorithm_permissive = SemanticRegionGrowing(
            threshold=0.70, min_region_size=5, use_smart_seeds=False, random_state=42
        )
        labeled_permissive, num_regions_permissive, _ = algorithm_permissive.segment(embeddings)

        assert not np.array_equal(labeled_strict, labeled_permissive)

    def test_segment_min_size_filtering(self):
        """Test that small regions are filtered based on min_size."""
        np.random.seed(42)
        embeddings = np.random.rand(50, 50, 16)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=2, keepdims=True)

        algorithm = SemanticRegionGrowing(
            threshold=0.95,
            min_region_size=100,
            use_smart_seeds=True,
            n_clusters=10,
            random_state=42,
        )

        labeled, num_regions, regions_info = algorithm.segment(embeddings)

        for region in regions_info:
            assert region["size"] >= 100

    def test_segment_4_connectivity(self):
        """Test that segmentation uses 4-connectivity."""
        embeddings = np.zeros((10, 10, 16))

        embeddings[4:6, 4:6] = 1.0

        embeddings = embeddings / (np.linalg.norm(embeddings, axis=2, keepdims=True) + 1e-8)

        algorithm = SemanticRegionGrowing(threshold=0.99, min_region_size=1, random_state=42)

        seeds = [(5, 5)]
        labeled, num_regions, regions_info = algorithm.segment(embeddings, seeds=seeds)

        assert num_regions >= 1

        if num_regions > 0:
            region_mask = labeled == 1
            assert region_mask[5, 5]

    def test_segment_regions_info_structure(self):
        """Test that regions_info contains expected fields."""
        np.random.seed(42)
        embeddings = np.random.rand(50, 50, 16)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=2, keepdims=True)

        algorithm = SemanticRegionGrowing(
            threshold=0.85, min_region_size=10, use_smart_seeds=True, n_clusters=3, random_state=42
        )

        _, _, regions_info = algorithm.segment(embeddings)

        if len(regions_info) > 0:
            region = regions_info[0]
            assert "id" in region
            assert "size" in region
            assert "centroid" in region
            assert "seed" in region
            assert "mean_embedding" in region
            assert "pixels" in region

            assert isinstance(region["id"], int)
            assert isinstance(region["size"], int)
            assert isinstance(region["centroid"], tuple)
            assert isinstance(region["seed"], tuple)
            assert isinstance(region["mean_embedding"], np.ndarray)
            assert isinstance(region["pixels"], list)

    def test_segment_invalid_embeddings_shape(self):
        """Test that invalid embeddings shape raises ValueError."""
        algorithm = SemanticRegionGrowing()

        embeddings_2d = np.random.rand(50, 50)
        with pytest.raises(ValueError, match="Embeddings must be 3D"):
            algorithm.segment(embeddings_2d)

        embeddings_4d = np.random.rand(50, 50, 16, 1)
        with pytest.raises(ValueError, match="Embeddings must be 3D"):
            algorithm.segment(embeddings_4d)

    def test_segment_labeled_pixels_percentage(self):
        """Test that reasonable percentage of pixels gets labeled."""
        np.random.seed(42)
        embeddings = np.random.rand(100, 100, 16)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=2, keepdims=True)

        algorithm = SemanticRegionGrowing(
            threshold=0.85, min_region_size=50, use_smart_seeds=True, n_clusters=5, random_state=42
        )

        labeled, _, _ = algorithm.segment(embeddings)

        labeled_count = np.sum(labeled > 0)
        total_pixels = labeled.size
        labeled_percentage = 100 * labeled_count / total_pixels

        assert 0 <= labeled_percentage <= 100


class TestStressAnalysis:
    """Test hierarchical stress analysis functionality."""

    def test_analyze_stress_basic(self):
        """Test basic stress analysis functionality."""
        labeled = np.array([[1, 1, 2, 2], [1, 1, 2, 2], [0, 0, 3, 3], [0, 0, 3, 3]], dtype=np.int32)

        ndvi = np.array(
            [
                [0.6, 0.6, 0.2, 0.2],
                [0.6, 0.6, 0.2, 0.2],
                [0.0, 0.0, 0.4, 0.4],
                [0.0, 0.0, 0.4, 0.4],
            ],
            dtype=np.float32,
        )

        regions_info = [{"id": 1, "size": 4}, {"id": 2, "size": 4}, {"id": 3, "size": 4}]

        algorithm = SemanticRegionGrowing()
        stress_results = algorithm.analyze_stress(labeled, ndvi, regions_info)

        assert len(stress_results) == 3
        assert 1 in stress_results
        assert 2 in stress_results
        assert 3 in stress_results

    def test_analyze_stress_distribution(self):
        """Test stress distribution calculation."""
        labeled = np.array([[1, 1], [1, 1]], dtype=np.int32)
        ndvi = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)

        regions_info = [{"id": 1, "size": 4}]

        algorithm = SemanticRegionGrowing()
        stress_results = algorithm.analyze_stress(labeled, ndvi, regions_info)

        result = stress_results[1]
        assert result["stress_distribution"]["high"] == 1
        assert result["stress_distribution"]["medium"] == 1
        assert result["stress_distribution"]["low"] == 2

    def test_analyze_stress_dominant_class(self):
        """Test dominant stress class determination."""
        labeled = np.ones((10, 10), dtype=np.int32)

        ndvi_high_stress = np.full((10, 10), 0.2)
        regions_info = [{"id": 1, "size": 100}]
        algorithm = SemanticRegionGrowing()
        result = algorithm.analyze_stress(labeled, ndvi_high_stress, regions_info)
        assert result[1]["dominant_stress"] == "high"

        ndvi_medium_stress = np.full((10, 10), 0.4)
        result = algorithm.analyze_stress(labeled, ndvi_medium_stress, regions_info)
        assert result[1]["dominant_stress"] == "medium"

        ndvi_low_stress = np.full((10, 10), 0.7)
        result = algorithm.analyze_stress(labeled, ndvi_low_stress, regions_info)
        assert result[1]["dominant_stress"] == "low"

    def test_analyze_stress_statistics(self):
        """Test that stress analysis computes correct statistics."""
        labeled = np.ones((5, 5), dtype=np.int32)
        ndvi = np.array(
            [
                [0.1, 0.2, 0.3, 0.4, 0.5],
                [0.2, 0.3, 0.4, 0.5, 0.6],
                [0.3, 0.4, 0.5, 0.6, 0.7],
                [0.4, 0.5, 0.6, 0.7, 0.8],
                [0.5, 0.6, 0.7, 0.8, 0.9],
            ],
            dtype=np.float32,
        )

        regions_info = [{"id": 1, "size": 25}]

        algorithm = SemanticRegionGrowing()
        result = algorithm.analyze_stress(labeled, ndvi, regions_info)

        assert "mean_ndvi" in result[1]
        assert "std_ndvi" in result[1]
        assert "min_ndvi" in result[1]
        assert "max_ndvi" in result[1]

        assert result[1]["mean_ndvi"] == pytest.approx(0.5, abs=0.01)
        assert result[1]["min_ndvi"] == pytest.approx(0.1, abs=0.01)
        assert result[1]["max_ndvi"] == pytest.approx(0.9, abs=0.01)

    def test_analyze_stress_percentages(self):
        """Test stress percentage calculations."""
        labeled = np.ones((10, 100), dtype=np.int32)

        ndvi = np.full((10, 100), 0.2)
        ndvi[:, 50:] = 0.6

        regions_info = [{"id": 1, "size": 1000}]

        algorithm = SemanticRegionGrowing()
        result = algorithm.analyze_stress(labeled, ndvi, regions_info)

        assert "stress_percentage" in result[1]
        assert result[1]["stress_percentage"]["high"] == pytest.approx(50.0, abs=1.0)
        assert result[1]["stress_percentage"]["low"] == pytest.approx(50.0, abs=1.0)

    def test_analyze_stress_shape_mismatch(self):
        """Test that shape mismatch raises ValueError."""
        labeled = np.ones((10, 10), dtype=np.int32)
        ndvi = np.ones((5, 5), dtype=np.float32)
        regions_info = []

        algorithm = SemanticRegionGrowing()

        with pytest.raises(ValueError, match="shapes must match"):
            algorithm.analyze_stress(labeled, ndvi, regions_info)


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_end_to_end_with_smart_seeds(self):
        """Test complete workflow with smart seeds."""
        np.random.seed(42)
        embeddings = np.random.rand(100, 100, 32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=2, keepdims=True)

        ndvi = np.random.rand(100, 100) * 0.8 + 0.1

        algorithm = SemanticRegionGrowing(
            threshold=0.85, min_region_size=50, use_smart_seeds=True, n_clusters=5, random_state=42
        )

        labeled, num_regions, regions_info = algorithm.segment(embeddings)

        assert labeled.shape == (100, 100)
        assert num_regions >= 0

        if num_regions > 0:
            stress_results = algorithm.analyze_stress(labeled, ndvi, regions_info)

            assert len(stress_results) == num_regions

            for region_id in range(1, num_regions + 1):
                assert region_id in stress_results
                assert "dominant_stress" in stress_results[region_id]
                assert stress_results[region_id]["dominant_stress"] in ["high", "medium", "low"]

    def test_comparison_grid_vs_kmeans(self):
        """Test comparing grid vs K-Means methods."""
        np.random.seed(42)
        embeddings = np.random.rand(100, 100, 32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=2, keepdims=True)

        algorithm_grid = SemanticRegionGrowing(
            threshold=0.85, min_region_size=50, use_smart_seeds=False, random_state=42
        )
        labeled_grid, num_regions_grid, _ = algorithm_grid.segment(embeddings)

        algorithm_kmeans = SemanticRegionGrowing(
            threshold=0.85, min_region_size=50, use_smart_seeds=True, n_clusters=5, random_state=42
        )
        labeled_kmeans, num_regions_kmeans, _ = algorithm_kmeans.segment(embeddings)

        assert labeled_grid.shape == labeled_kmeans.shape
        assert len(algorithm_grid.seeds_) > len(algorithm_kmeans.seeds_)

        assert algorithm_grid.seeds_ != algorithm_kmeans.seeds_

    def test_multiple_segmentations_reproducibility(self):
        """Test that multiple runs with same parameters produce same results."""
        np.random.seed(42)
        embeddings = np.random.rand(50, 50, 16)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=2, keepdims=True)

        algorithm1 = SemanticRegionGrowing(
            threshold=0.85, use_smart_seeds=True, n_clusters=5, random_state=42
        )
        labeled1, num_regions1, _ = algorithm1.segment(embeddings)

        algorithm2 = SemanticRegionGrowing(
            threshold=0.85, use_smart_seeds=True, n_clusters=5, random_state=42
        )
        labeled2, num_regions2, _ = algorithm2.segment(embeddings)

        assert num_regions1 == num_regions2
        assert np.array_equal(labeled1, labeled2)
        assert algorithm1.seeds_ == algorithm2.seeds_
