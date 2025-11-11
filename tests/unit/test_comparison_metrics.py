"""
Unit tests for comparison metrics module.
"""

import pytest
import numpy as np
from src.utils.comparison_metrics import (
    calculate_spatial_coherence,
    count_regions,
    calculate_region_statistics,
    compare_segmentations,
    calculate_boundary_precision,
    SegmentationMetrics,
)


class TestSpatialCoherence:
    """Tests for spatial coherence calculation."""

    def test_full_coverage(self):
        """Test with 100% labeled pixels."""
        seg = np.ones((100, 100), dtype=int)
        coherence = calculate_spatial_coherence(seg)
        assert coherence == 100.0

    def test_half_coverage(self):
        """Test with 50% labeled pixels."""
        seg = np.zeros((100, 100), dtype=int)
        seg[:50, :] = 1
        coherence = calculate_spatial_coherence(seg)
        assert coherence == 50.0

    def test_no_coverage(self):
        """Test with 0% labeled pixels."""
        seg = np.zeros((100, 100), dtype=int)
        coherence = calculate_spatial_coherence(seg)
        assert coherence == 0.0

    def test_partial_coverage(self):
        """Test with partial coverage."""
        seg = np.zeros((100, 100), dtype=int)
        seg[25:75, 25:75] = 1
        coherence = calculate_spatial_coherence(seg)
        expected = (50 * 50) / (100 * 100) * 100
        assert coherence == pytest.approx(expected, rel=1e-5)

    def test_empty_segmentation(self):
        """Test with empty segmentation."""
        seg = np.array([], dtype=int)
        coherence = calculate_spatial_coherence(seg)
        assert coherence == 0.0


class TestCountRegions:
    """Tests for region counting."""

    def test_single_region(self):
        """Test with single region."""
        seg = np.ones((100, 100), dtype=int)
        count = count_regions(seg)
        assert count == 1

    def test_multiple_regions(self):
        """Test with multiple regions."""
        seg = np.zeros((100, 100), dtype=int)
        seg[:50, :] = 1
        seg[50:, :50] = 2
        seg[50:, 50:] = 3
        count = count_regions(seg)
        assert count == 3

    def test_with_background(self):
        """Test that background (0) is excluded."""
        seg = np.zeros((100, 100), dtype=int)
        seg[25:75, 25:75] = 1
        count = count_regions(seg)
        assert count == 1

    def test_no_regions(self):
        """Test with only background."""
        seg = np.zeros((100, 100), dtype=int)
        count = count_regions(seg)
        assert count == 0

    def test_many_regions(self):
        """Test with many regions."""
        seg = np.arange(1, 101).reshape((10, 10))
        count = count_regions(seg)
        assert count == 100


class TestRegionStatistics:
    """Tests for region statistics calculation."""

    def test_uniform_regions(self):
        """Test with uniform region sizes."""
        seg = np.zeros((100, 100), dtype=int)
        seg[:50, :] = 1
        seg[50:, :] = 2
        stats = calculate_region_statistics(seg)

        assert stats["avg_size"] == 5000.0
        assert stats["std_size"] == 0.0
        assert stats["largest_size"] == 5000
        assert stats["smallest_size"] == 5000

    def test_varied_regions(self):
        """Test with varied region sizes."""
        seg = np.zeros((100, 100), dtype=int)
        seg[:25, :] = 1  # 2500 pixels
        seg[25:75, :] = 2  # 5000 pixels
        seg[75:, :] = 3  # 2500 pixels
        stats = calculate_region_statistics(seg)

        expected_avg = (2500 + 5000 + 2500) / 3
        assert stats["avg_size"] == pytest.approx(expected_avg, rel=1e-5)
        assert stats["largest_size"] == 5000
        assert stats["smallest_size"] == 2500

    def test_no_regions(self):
        """Test with no regions."""
        seg = np.zeros((100, 100), dtype=int)
        stats = calculate_region_statistics(seg)

        assert stats["avg_size"] == 0.0
        assert stats["std_size"] == 0.0
        assert stats["largest_size"] == 0
        assert stats["smallest_size"] == 0

    def test_single_pixel_regions(self):
        """Test with single pixel regions."""
        seg = np.arange(1, 101).reshape((10, 10))
        stats = calculate_region_statistics(seg)

        assert stats["avg_size"] == 1.0
        assert stats["std_size"] == 0.0
        assert stats["largest_size"] == 1
        assert stats["smallest_size"] == 1


class TestCompareSegmentations:
    """Tests for segmentation comparison."""

    def test_basic_comparison(self):
        """Test basic comparison."""
        classic_seg = np.random.randint(0, 10, (100, 100))
        mgrg_seg = np.random.randint(0, 5, (100, 100))

        metrics = compare_segmentations(classic_seg, mgrg_seg, 1.0, 1.5)

        assert "classic" in metrics
        assert "mgrg" in metrics
        assert "differences" in metrics
        assert "winner" in metrics
        assert isinstance(metrics["classic"], SegmentationMetrics)
        assert isinstance(metrics["mgrg"], SegmentationMetrics)

    def test_shape_mismatch(self):
        """Test with mismatched shapes."""
        classic_seg = np.ones((100, 100), dtype=int)
        mgrg_seg = np.ones((50, 50), dtype=int)

        with pytest.raises(ValueError, match="do not match"):
            compare_segmentations(classic_seg, mgrg_seg, 1.0, 1.5)

    def test_difference_calculations(self):
        """Test difference calculations."""
        classic_seg = np.ones((100, 100), dtype=int)
        mgrg_seg = np.ones((100, 100), dtype=int) * 2

        metrics = compare_segmentations(classic_seg, mgrg_seg, 1.0, 1.5)

        assert metrics["differences"]["num_regions"] == 0  # Both have 1 region
        assert metrics["differences"]["coherence"] == 0.0  # Both 100%
        assert metrics["differences"]["time"] == 0.5

    def test_winner_determination(self):
        """Test winner determination based on coherence."""
        classic_seg = np.zeros((100, 100), dtype=int)
        classic_seg[:50, :] = 1  # 50% coherence

        mgrg_seg = np.ones((100, 100), dtype=int)  # 100% coherence

        metrics = compare_segmentations(classic_seg, mgrg_seg, 1.0, 1.5)

        assert metrics["winner"] == "mgrg"


class TestBoundaryPrecision:
    """Tests for boundary precision calculation."""

    def test_perfect_match(self):
        """Test with perfect match."""
        pred = np.ones((100, 100), dtype=bool)
        gt = np.ones((100, 100), dtype=bool)
        iou = calculate_boundary_precision(pred, gt)
        assert iou == 1.0

    def test_no_overlap(self):
        """Test with no overlap."""
        pred = np.zeros((100, 100), dtype=bool)
        pred[:50, :] = True
        gt = np.zeros((100, 100), dtype=bool)
        gt[50:, :] = True
        iou = calculate_boundary_precision(pred, gt)
        assert iou == 0.0

    def test_partial_overlap(self):
        """Test with partial overlap."""
        pred = np.zeros((100, 100), dtype=bool)
        pred[:75, :] = True
        gt = np.zeros((100, 100), dtype=bool)
        gt[:50, :] = True

        iou = calculate_boundary_precision(pred, gt)
        # Intersection: 50*100 = 5000
        # Union: 75*100 = 7500
        # IoU: 5000/7500 = 0.6667
        expected = 5000 / 7500
        assert iou == pytest.approx(expected, rel=1e-5)

    def test_shape_mismatch(self):
        """Test with mismatched shapes."""
        pred = np.ones((100, 100), dtype=bool)
        gt = np.ones((50, 50), dtype=bool)

        with pytest.raises(ValueError, match="Shape mismatch"):
            calculate_boundary_precision(pred, gt)

    def test_empty_union(self):
        """Test with empty union."""
        pred = np.zeros((100, 100), dtype=bool)
        gt = np.zeros((100, 100), dtype=bool)
        iou = calculate_boundary_precision(pred, gt)
        assert iou == 0.0


class TestSegmentationMetrics:
    """Tests for SegmentationMetrics dataclass."""

    def test_dataclass_creation(self):
        """Test creating SegmentationMetrics."""
        metrics = SegmentationMetrics(
            num_regions=10,
            coherence=85.5,
            avg_region_size=1000.0,
            std_region_size=200.0,
            largest_region_size=1500,
            smallest_region_size=500,
            processing_time=1.5,
        )

        assert metrics.num_regions == 10
        assert metrics.coherence == 85.5
        assert metrics.avg_region_size == 1000.0
        assert metrics.std_region_size == 200.0
        assert metrics.largest_region_size == 1500
        assert metrics.smallest_region_size == 500
        assert metrics.processing_time == 1.5

    def test_dataclass_attributes(self):
        """Test dataclass has all required attributes."""
        metrics = SegmentationMetrics(
            num_regions=5,
            coherence=90.0,
            avg_region_size=500.0,
            std_region_size=100.0,
            largest_region_size=800,
            smallest_region_size=200,
            processing_time=2.0,
        )

        assert hasattr(metrics, "num_regions")
        assert hasattr(metrics, "coherence")
        assert hasattr(metrics, "avg_region_size")
        assert hasattr(metrics, "std_region_size")
        assert hasattr(metrics, "largest_region_size")
        assert hasattr(metrics, "smallest_region_size")
        assert hasattr(metrics, "processing_time")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
