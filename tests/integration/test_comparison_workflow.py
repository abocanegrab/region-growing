"""
Integration tests for full comparison workflow.

Tests the complete pipeline from data generation through metrics
calculation to visualization export.
"""

import pytest
import numpy as np
from pathlib import Path

from src.utils.comparison_metrics import (
    compare_segmentations,
    SegmentationMetrics,
)
from src.visualization.ab_comparison import (
    create_side_by_side_plot,
    create_metrics_table,
    export_high_resolution,
    generate_failure_case_analysis,
)


class TestComparisonWorkflow:
    """Integration tests for full comparison workflow."""

    def test_end_to_end_comparison(self, tmp_path):
        """Test complete comparison workflow."""
        # Setup: Generate synthetic data
        rgb = np.random.rand(200, 200, 3)
        ndvi = np.random.rand(200, 200) * 2 - 1

        # Generate synthetic segmentations
        # Classic: more regions, lower coherence
        classic_seg = np.random.randint(0, 15, (200, 200))
        # MGRG: fewer regions, higher coherence
        mgrg_seg = np.random.randint(0, 5, (200, 200))

        # Ensure some unlabeled pixels for coherence calculation
        classic_seg[classic_seg > 12] = 0
        mgrg_seg[mgrg_seg > 3] = 0

        # Calculate metrics
        metrics = compare_segmentations(
            classic_seg, mgrg_seg, 1.0, 1.5
        )

        # Assertions on metrics
        assert "classic" in metrics
        assert "mgrg" in metrics
        assert "differences" in metrics
        assert "winner" in metrics
        assert isinstance(metrics["classic"], SegmentationMetrics)
        assert isinstance(metrics["mgrg"], SegmentationMetrics)
        assert metrics["classic"].num_regions > 0
        assert metrics["mgrg"].num_regions > 0

        # Create visualization
        save_path = tmp_path / "comparison.png"
        fig, img = create_side_by_side_plot(
            rgb, classic_seg, mgrg_seg, metrics,
            save_path=str(save_path)
        )

        # Assertions on visualization
        assert save_path.exists()
        assert img.shape[2] == 3  # RGB
        assert img.dtype == np.uint8

    def test_metrics_to_visualization_workflow(self, tmp_path):
        """Test workflow from metrics calculation to table visualization."""
        # Generate synthetic segmentations
        classic_seg = np.random.randint(0, 10, (200, 200))
        mgrg_seg = np.random.randint(0, 5, (200, 200))

        # Calculate metrics
        metrics = compare_segmentations(
            classic_seg, mgrg_seg, 1.2, 1.4
        )

        # Create metrics table
        save_path = tmp_path / "metrics_table.png"
        fig = create_metrics_table(metrics, save_path=str(save_path))

        # Assertions
        assert save_path.exists()
        assert fig is not None

    def test_failure_case_workflow(self, tmp_path):
        """Test workflow for failure case analysis."""
        # Generate synthetic data
        rgb = np.random.rand(200, 200, 3)
        ndvi = np.random.rand(200, 200) * 2 - 1
        classic_seg = np.random.randint(0, 20, (200, 200))
        mgrg_seg = np.random.randint(0, 3, (200, 200))

        # Generate failure case analysis
        save_dir = tmp_path / "failure_cases"
        path = generate_failure_case_analysis(
            zone_name="test_zone",
            rgb_image=rgb,
            classic_seg=classic_seg,
            mgrg_seg=mgrg_seg,
            ndvi=ndvi,
            failure_description="Test failure case",
            save_dir=str(save_dir)
        )

        # Assertions
        assert Path(path).exists()
        assert "test_zone" in path

    def test_multi_format_export_workflow(self, tmp_path):
        """Test workflow with multiple export formats."""
        # Generate synthetic data
        rgb = np.random.rand(100, 100, 3)
        classic_seg = np.random.randint(0, 10, (100, 100))
        mgrg_seg = np.random.randint(0, 5, (100, 100))

        # Calculate metrics
        metrics = compare_segmentations(
            classic_seg, mgrg_seg, 1.0, 1.5
        )

        # Create visualization
        base_path = tmp_path / "comparison"
        fig, _ = create_side_by_side_plot(
            rgb, classic_seg, mgrg_seg, metrics
        )

        # Export in multiple formats
        exported = export_high_resolution(
            fig, str(base_path), dpi=150, formats=['png', 'pdf']
        )

        # Assertions
        assert 'png' in exported
        assert 'pdf' in exported
        assert Path(exported['png']).exists()
        assert Path(exported['pdf']).exists()

    def test_batch_comparison_workflow(self, tmp_path):
        """Test workflow for batch comparison of multiple zones."""
        zones = ['zone1', 'zone2', 'zone3']
        results = {}

        for zone in zones:
            # Generate synthetic data for each zone
            classic_seg = np.random.randint(0, 15, (150, 150))
            mgrg_seg = np.random.randint(0, 5, (150, 150))

            # Calculate metrics
            metrics = compare_segmentations(
                classic_seg, mgrg_seg, 1.1, 1.3
            )
            results[zone] = metrics

            # Create visualization
            rgb = np.random.rand(150, 150, 3)
            save_path = tmp_path / f"{zone}_comparison.png"
            fig, _ = create_side_by_side_plot(
                rgb, classic_seg, mgrg_seg, metrics,
                title=f"Comparison: {zone}",
                save_path=str(save_path)
            )

            # Assertions for this zone
            assert save_path.exists()

        # Assertions for batch
        assert len(results) == len(zones)
        for zone in zones:
            assert zone in results
            assert "winner" in results[zone]


class TestComparisonMetricsWorkflow:
    """Integration tests for metrics calculation workflow."""

    def test_metrics_consistency(self):
        """Test that metrics are consistent across multiple calculations."""
        # Generate segmentation
        seg = np.random.randint(0, 5, (100, 100))

        # Calculate metrics multiple times
        from src.utils.comparison_metrics import (
            calculate_spatial_coherence,
            count_regions,
            calculate_region_statistics
        )

        coherence1 = calculate_spatial_coherence(seg)
        coherence2 = calculate_spatial_coherence(seg)
        assert coherence1 == coherence2

        regions1 = count_regions(seg)
        regions2 = count_regions(seg)
        assert regions1 == regions2

        stats1 = calculate_region_statistics(seg)
        stats2 = calculate_region_statistics(seg)
        assert stats1 == stats2

    def test_metrics_with_edge_cases(self):
        """Test metrics with edge cases."""
        # Empty segmentation
        empty_seg = np.zeros((100, 100), dtype=int)
        from src.utils.comparison_metrics import (
            calculate_spatial_coherence,
            count_regions
        )

        coherence = calculate_spatial_coherence(empty_seg)
        assert coherence == 0.0

        regions = count_regions(empty_seg)
        assert regions == 0

        # Full segmentation
        full_seg = np.ones((100, 100), dtype=int)
        coherence = calculate_spatial_coherence(full_seg)
        assert coherence == 100.0

        regions = count_regions(full_seg)
        assert regions == 1


class TestVisualizationWorkflow:
    """Integration tests for visualization workflow."""

    def test_visualization_with_different_sizes(self, tmp_path):
        """Test visualization with different image sizes."""
        sizes = [(100, 100), (200, 150), (512, 512)]

        for h, w in sizes:
            rgb = np.random.rand(h, w, 3)
            seg1 = np.random.randint(0, 5, (h, w))
            seg2 = np.random.randint(0, 3, (h, w))

            metrics = compare_segmentations(seg1, seg2, 1.0, 1.0)

            save_path = tmp_path / f"comparison_{h}x{w}.png"
            fig, img = create_side_by_side_plot(
                rgb, seg1, seg2, metrics,
                save_path=str(save_path)
            )

            assert save_path.exists()
            assert img.shape[2] == 3

    def test_visualization_with_extreme_metrics(self, tmp_path):
        """Test visualization with extreme metric values."""
        # Case 1: Very low coherence
        seg_low = np.zeros((100, 100), dtype=int)
        seg_low[:10, :10] = 1

        # Case 2: Perfect coherence
        seg_high = np.ones((100, 100), dtype=int)

        metrics = compare_segmentations(seg_low, seg_high, 1.0, 1.0)

        rgb = np.random.rand(100, 100, 3)
        save_path = tmp_path / "extreme_metrics.png"
        fig, _ = create_side_by_side_plot(
            rgb, seg_low, seg_high, metrics,
            save_path=str(save_path)
        )

        assert save_path.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
