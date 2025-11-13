"""Integration tests for complete classification workflow."""

import pytest
import numpy as np
from pathlib import Path

from src.classification.zero_shot_classifier import (
    SemanticClassifier,
    cross_validate_with_dynamic_world,
    LAND_COVER_CLASSES
)


class TestClassificationWorkflow:
    """Integration tests for complete classification workflow"""

    @pytest.fixture
    def mexicali_data(self):
        """Load real Mexicali data for testing"""
        base_path = Path("data/processed/mexicali")

        # Load real data
        ndvi = np.load(base_path / "ndvi.npy")
        segmentation = np.load(base_path / "mgrg_segmentation.npy")

        # For embeddings, use synthetic ones if real ones not available
        embeddings_path = Path("data/processed/us-005/synthetic_embeddings.npy")
        if embeddings_path.exists():
            embeddings_full = np.load(embeddings_path)
            # Resize to match NDVI shape if needed
            if embeddings_full.shape[:2] != ndvi.shape:
                # Create synthetic embeddings matching NDVI shape
                h, w = ndvi.shape
                embeddings = np.random.rand(h, w, 256).astype(np.float32)
            else:
                embeddings = embeddings_full
        else:
            # Create synthetic embeddings if none available
            h, w = ndvi.shape
            embeddings = np.random.rand(h, w, 256).astype(np.float32)

        return {
            'embeddings': embeddings,
            'ndvi': ndvi,
            'segmentation': segmentation
        }

    def test_end_to_end_classification_workflow(self, mexicali_data):
        """Should complete full workflow from data to visualization"""
        embeddings = mexicali_data['embeddings']
        ndvi = mexicali_data['ndvi']
        segmentation = mexicali_data['segmentation']

        # Classify
        classifier = SemanticClassifier(embeddings, ndvi)
        results = classifier.classify_all_regions(segmentation)

        # Generate maps
        semantic_map = classifier.generate_semantic_map(segmentation, results)
        colored_map = classifier.generate_colored_map(semantic_map)

        # Statistics
        stats = classifier.get_class_statistics(results)

        # Assertions
        assert len(results) > 0, "Should classify at least one region"
        assert semantic_map.shape == segmentation.shape, "Semantic map shape mismatch"
        assert colored_map.shape == (*segmentation.shape, 3), "Colored map should be RGB"
        assert len(stats) == 6, "Should have stats for all 6 classes"

        # Verify semantic map contains valid class IDs
        unique_classes = np.unique(semantic_map)
        assert all(c <= 5 for c in unique_classes), "All class IDs should be 0-5"

        # Verify colored map has RGB values
        assert colored_map.dtype == np.uint8, "Colored map should be uint8"
        assert colored_map.min() >= 0 and colored_map.max() <= 255, "RGB values should be 0-255"

    def test_classification_with_real_data_consistency(self, mexicali_data):
        """Should produce consistent results with real data"""
        embeddings = mexicali_data['embeddings']
        ndvi = mexicali_data['ndvi']
        segmentation = mexicali_data['segmentation']

        # Run classification twice
        classifier = SemanticClassifier(embeddings, ndvi)
        results1 = classifier.classify_all_regions(segmentation)
        results2 = classifier.classify_all_regions(segmentation)

        # Results should be deterministic
        assert len(results1) == len(results2), "Should classify same number of regions"

        for region_id in results1:
            assert region_id in results2, f"Region {region_id} should be in both results"
            assert results1[region_id].class_id == results2[region_id].class_id, \
                f"Region {region_id} should have same class"
            assert abs(results1[region_id].mean_ndvi - results2[region_id].mean_ndvi) < 1e-5, \
                f"Region {region_id} should have same NDVI"

    def test_classification_performance(self, mexicali_data):
        """Should complete classification in reasonable time"""
        import time

        embeddings = mexicali_data['embeddings']
        ndvi = mexicali_data['ndvi']
        segmentation = mexicali_data['segmentation']

        classifier = SemanticClassifier(embeddings, ndvi)

        start = time.time()
        results = classifier.classify_all_regions(segmentation)
        elapsed = time.time() - start

        # Should complete in less than 10 seconds for typical dataset
        assert elapsed < 10.0, f"Classification took {elapsed:.2f}s, should be <10s"
        assert len(results) > 0, "Should classify at least one region"

    def test_semantic_map_covers_all_regions(self, mexicali_data):
        """Should ensure semantic map covers all segmented regions"""
        embeddings = mexicali_data['embeddings']
        ndvi = mexicali_data['ndvi']
        segmentation = mexicali_data['segmentation']

        classifier = SemanticClassifier(embeddings, ndvi)
        results = classifier.classify_all_regions(segmentation)
        semantic_map = classifier.generate_semantic_map(segmentation, results)

        # Count pixels in segmentation (excluding background)
        seg_pixels = (segmentation > 0).sum()

        # Count classified pixels in semantic map
        # Note: Some regions might be skipped due to min_size filter
        classified_pixels = (semantic_map > 0).sum() if seg_pixels > 0 else 0

        # At least 80% of segmented pixels should be classified
        # (some small regions may be filtered out)
        if seg_pixels > 0:
            coverage = classified_pixels / seg_pixels
            assert coverage > 0.8, \
                f"Coverage {coverage:.1%} too low, should classify >80% of pixels"

    def test_class_statistics_consistency(self, mexicali_data):
        """Should produce consistent class statistics"""
        embeddings = mexicali_data['embeddings']
        ndvi = mexicali_data['ndvi']
        segmentation = mexicali_data['segmentation']

        classifier = SemanticClassifier(embeddings, ndvi)
        results = classifier.classify_all_regions(segmentation)
        stats = classifier.get_class_statistics(results)

        # Check stats structure
        for class_name in LAND_COVER_CLASSES.values():
            assert class_name in stats, f"Missing stats for {class_name}"
            assert 'count' in stats[class_name], f"Missing count for {class_name}"
            assert 'area_ha' in stats[class_name], f"Missing area for {class_name}"
            assert 'mean_ndvi' in stats[class_name], f"Missing mean_ndvi for {class_name}"
            assert 'std_ndvi' in stats[class_name], f"Missing std_ndvi for {class_name}"

        # Total area should match total classified pixels
        total_area = sum(stats[c]['area_ha'] for c in LAND_COVER_CLASSES.values())
        total_pixels = sum(r.size_pixels for r in results.values())
        expected_area = total_pixels * (10 ** 2) / 10000  # 10m resolution to hectares

        assert abs(total_area - expected_area) < 0.1, \
            f"Total area {total_area:.2f} ha doesn't match expected {expected_area:.2f} ha"

    def test_cross_validation_with_synthetic_dynamic_world(self, mexicali_data):
        """Should run cross-validation workflow with synthetic DW data"""
        embeddings = mexicali_data['embeddings']
        ndvi = mexicali_data['ndvi']
        segmentation = mexicali_data['segmentation']

        # Classify
        classifier = SemanticClassifier(embeddings, ndvi)
        results = classifier.classify_all_regions(segmentation)
        semantic_map = classifier.generate_semantic_map(segmentation, results)

        # Create synthetic Dynamic World mask for testing
        # In real scenario, this would be loaded from data/dynamic_world/
        h, w = semantic_map.shape
        dw_mask = np.zeros((h, w), dtype=np.uint8)

        # Create some synthetic DW data based on semantic map
        # Add some noise to simulate imperfect agreement
        for class_id in range(6):
            mask = (semantic_map == class_id)
            if mask.any():
                # Map our classes to DW classes with some noise
                if class_id == 0:  # Water → DW Water (0)
                    dw_mask[mask] = 0
                elif class_id == 1:  # Urban → DW Built (6)
                    dw_mask[mask] = 6
                elif class_id == 2:  # Bare Soil → DW Bare (7)
                    dw_mask[mask] = 7
                elif class_id in [3, 4]:  # Crops → DW Crops (4)
                    dw_mask[mask] = 4
                elif class_id == 5:  # Grass → DW Grass (2)
                    dw_mask[mask] = 2

        # Run cross-validation
        agreements = cross_validate_with_dynamic_world(semantic_map, dw_mask)

        # Check results
        assert 'overall' in agreements, "Should have overall agreement"
        assert 0.0 <= agreements['overall'] <= 1.0, "Overall agreement should be 0-1"

        # With synthetic matching data, agreement should be very high
        assert agreements['overall'] > 0.9, \
            f"Agreement {agreements['overall']:.1%} too low for synthetic matching data"

    def test_classification_handles_edge_cases(self, mexicali_data):
        """Should handle edge cases gracefully"""
        embeddings = mexicali_data['embeddings']
        ndvi = mexicali_data['ndvi']

        # Create segmentation with edge cases
        h, w = ndvi.shape
        segmentation = np.zeros((h, w), dtype=np.int32)

        # Very small region (should be filtered out)
        segmentation[0:2, 0:2] = 1  # 4 pixels

        # Normal-sized region
        segmentation[10:50, 10:50] = 2  # 1600 pixels

        # Large region
        segmentation[100:200, 100:300] = 3  # 20000 pixels

        classifier = SemanticClassifier(embeddings, ndvi)
        results = classifier.classify_all_regions(segmentation, min_size=10)

        # Small region should be filtered out
        assert 1 not in results, "Very small region should be filtered"

        # Other regions should be classified
        assert 2 in results, "Normal region should be classified"
        assert 3 in results, "Large region should be classified"

        # All classified regions should have valid data
        for region_id, result in results.items():
            assert result.class_id in range(6), f"Region {region_id} has invalid class"
            assert result.size_pixels >= 10, f"Region {region_id} too small"
            assert result.area_hectares > 0, f"Region {region_id} has zero area"
            assert -1 <= result.mean_ndvi <= 1, f"Region {region_id} has invalid NDVI"
