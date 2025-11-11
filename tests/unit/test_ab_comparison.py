"""
Unit tests for A/B comparison visualization module.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.visualization.ab_comparison import (
    create_side_by_side_plot,
    create_metrics_table,
    create_overlay_comparison,
    export_high_resolution,
    generate_failure_case_analysis,
)
from src.utils.comparison_metrics import SegmentationMetrics


@pytest.fixture
def sample_rgb():
    """Sample RGB image."""
    return np.random.rand(100, 100, 3)


@pytest.fixture
def sample_segmentation1():
    """Sample segmentation 1."""
    return np.random.randint(0, 5, (100, 100))


@pytest.fixture
def sample_segmentation2():
    """Sample segmentation 2."""
    return np.random.randint(0, 3, (100, 100))


@pytest.fixture
def sample_ndvi():
    """Sample NDVI array."""
    return np.random.rand(100, 100) * 2 - 1  # Range -1 to 1


@pytest.fixture
def sample_metrics():
    """Sample metrics dictionary."""
    return {
        "classic": SegmentationMetrics(
            num_regions=10,
            coherence=80.0,
            avg_region_size=1000.0,
            std_region_size=200.0,
            largest_region_size=1500,
            smallest_region_size=500,
            processing_time=1.0,
        ),
        "mgrg": SegmentationMetrics(
            num_regions=5,
            coherence=95.0,
            avg_region_size=2000.0,
            std_region_size=300.0,
            largest_region_size=2500,
            smallest_region_size=1500,
            processing_time=1.5,
        ),
        "differences": {
            "num_regions": -5,
            "coherence": 15.0,
            "avg_size": 1000.0,
            "time": 0.5,
        },
        "winner": "mgrg",
    }


class TestSideBySidePlot:
    """Tests for side-by-side comparison plot."""

    def test_creates_figure(
        self, sample_rgb, sample_segmentation1, sample_segmentation2, sample_metrics
    ):
        """Test that function creates valid figure."""
        fig, img = create_side_by_side_plot(
            sample_rgb, sample_segmentation1, sample_segmentation2, sample_metrics
        )

        assert isinstance(fig, plt.Figure)
        assert img.shape[2] == 3  # RGB
        assert img.dtype == np.uint8
        plt.close(fig)

    def test_figure_has_subplots(
        self, sample_rgb, sample_segmentation1, sample_segmentation2, sample_metrics
    ):
        """Test that figure has correct number of subplots."""
        fig, _ = create_side_by_side_plot(
            sample_rgb, sample_segmentation1, sample_segmentation2, sample_metrics
        )

        axes = fig.get_axes()
        assert len(axes) == 6  # 2x3 grid
        plt.close(fig)

    def test_saves_to_file(
        self,
        tmp_path,
        sample_rgb,
        sample_segmentation1,
        sample_segmentation2,
        sample_metrics,
    ):
        """Test that function saves to file."""
        save_path = tmp_path / "test_comparison.png"
        fig, _ = create_side_by_side_plot(
            sample_rgb,
            sample_segmentation1,
            sample_segmentation2,
            sample_metrics,
            save_path=str(save_path),
        )

        assert save_path.exists()
        plt.close(fig)

    def test_custom_title(
        self, sample_rgb, sample_segmentation1, sample_segmentation2, sample_metrics
    ):
        """Test with custom title."""
        custom_title = "Custom Comparison Title"
        fig, _ = create_side_by_side_plot(
            sample_rgb,
            sample_segmentation1,
            sample_segmentation2,
            sample_metrics,
            title=custom_title,
        )

        assert fig._suptitle.get_text() == custom_title
        plt.close(fig)

    def test_custom_dpi(
        self,
        tmp_path,
        sample_rgb,
        sample_segmentation1,
        sample_segmentation2,
        sample_metrics,
    ):
        """Test with custom DPI."""
        save_path = tmp_path / "test_comparison_high_dpi.png"
        fig, _ = create_side_by_side_plot(
            sample_rgb,
            sample_segmentation1,
            sample_segmentation2,
            sample_metrics,
            save_path=str(save_path),
            dpi=600,
        )

        assert save_path.exists()
        plt.close(fig)


class TestMetricsTable:
    """Tests for metrics table creation."""

    def test_creates_figure(self, sample_metrics):
        """Test that function creates valid figure."""
        fig = create_metrics_table(sample_metrics)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_to_file(self, tmp_path, sample_metrics):
        """Test that function saves to file."""
        save_path = tmp_path / "test_metrics_table.png"
        fig = create_metrics_table(sample_metrics, save_path=str(save_path))

        assert save_path.exists()
        plt.close(fig)

    def test_table_has_title(self, sample_metrics):
        """Test that table has title."""
        fig = create_metrics_table(sample_metrics)

        assert fig._suptitle is not None
        plt.close(fig)


class TestOverlayComparison:
    """Tests for overlay comparison."""

    def test_creates_overlays(
        self, sample_rgb, sample_segmentation1, sample_segmentation2
    ):
        """Test that function creates overlays."""
        overlay_classic, overlay_mgrg = create_overlay_comparison(
            sample_rgb, sample_segmentation1, sample_segmentation2
        )

        assert overlay_classic.shape == sample_rgb.shape
        assert overlay_mgrg.shape == sample_rgb.shape
        assert overlay_classic.dtype == np.uint8
        assert overlay_mgrg.dtype == np.uint8

    def test_custom_alpha(
        self, sample_rgb, sample_segmentation1, sample_segmentation2
    ):
        """Test with custom alpha value."""
        overlay_classic, overlay_mgrg = create_overlay_comparison(
            sample_rgb, sample_segmentation1, sample_segmentation2, alpha=0.3
        )

        assert overlay_classic.shape == sample_rgb.shape
        assert overlay_mgrg.shape == sample_rgb.shape

    def test_normalized_rgb(self, sample_segmentation1, sample_segmentation2):
        """Test with normalized RGB (0-1)."""
        rgb_normalized = np.random.rand(100, 100, 3)
        overlay_classic, overlay_mgrg = create_overlay_comparison(
            rgb_normalized, sample_segmentation1, sample_segmentation2
        )

        assert overlay_classic.dtype == np.uint8
        assert overlay_mgrg.dtype == np.uint8


class TestExportHighResolution:
    """Tests for high-resolution export."""

    def test_exports_single_format(self, tmp_path):
        """Test export in single format."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        base_path = tmp_path / "test_export"
        exported = export_high_resolution(
            fig, str(base_path), dpi=300, formats=["png"]
        )

        assert "png" in exported
        assert Path(exported["png"]).exists()
        plt.close(fig)

    def test_exports_multiple_formats(self, tmp_path):
        """Test export in multiple formats."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        base_path = tmp_path / "test_export"
        exported = export_high_resolution(
            fig, str(base_path), dpi=300, formats=["png", "pdf", "svg"]
        )

        assert "png" in exported
        assert "pdf" in exported
        assert "svg" in exported
        assert Path(exported["png"]).exists()
        assert Path(exported["pdf"]).exists()
        assert Path(exported["svg"]).exists()
        plt.close(fig)

    def test_creates_parent_directory(self, tmp_path):
        """Test that parent directory is created."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        base_path = tmp_path / "nested" / "dir" / "test_export"
        exported = export_high_resolution(
            fig, str(base_path), dpi=300, formats=["png"]
        )

        assert Path(exported["png"]).exists()
        plt.close(fig)

    def test_custom_dpi(self, tmp_path):
        """Test with custom DPI."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        base_path = tmp_path / "test_export_dpi"
        exported = export_high_resolution(
            fig, str(base_path), dpi=600, formats=["png"]
        )

        assert Path(exported["png"]).exists()
        plt.close(fig)


class TestFailureCaseAnalysis:
    """Tests for failure case analysis."""

    def test_creates_analysis(
        self,
        tmp_path,
        sample_rgb,
        sample_segmentation1,
        sample_segmentation2,
        sample_ndvi,
    ):
        """Test that function creates analysis."""
        zone_name = "test_zone"
        description = "Test failure case description"
        save_dir = tmp_path / "failure_cases"

        path = generate_failure_case_analysis(
            zone_name,
            sample_rgb,
            sample_segmentation1,
            sample_segmentation2,
            sample_ndvi,
            description,
            str(save_dir),
        )

        assert Path(path).exists()
        assert zone_name in path

    def test_creates_save_directory(
        self,
        tmp_path,
        sample_rgb,
        sample_segmentation1,
        sample_segmentation2,
        sample_ndvi,
    ):
        """Test that save directory is created."""
        zone_name = "test_zone"
        description = "Test description"
        save_dir = tmp_path / "new_failure_dir"

        path = generate_failure_case_analysis(
            zone_name,
            sample_rgb,
            sample_segmentation1,
            sample_segmentation2,
            sample_ndvi,
            description,
            str(save_dir),
        )

        assert Path(save_dir).exists()
        assert Path(path).exists()

    def test_custom_zone_name(
        self,
        tmp_path,
        sample_rgb,
        sample_segmentation1,
        sample_segmentation2,
        sample_ndvi,
    ):
        """Test with custom zone name."""
        zone_name = "mexicali_cloud_shadow"
        description = "Cloud shadow causes fragmentation"
        save_dir = tmp_path / "failure_cases"

        path = generate_failure_case_analysis(
            zone_name,
            sample_rgb,
            sample_segmentation1,
            sample_segmentation2,
            sample_ndvi,
            description,
            str(save_dir),
        )

        assert "mexicali_cloud_shadow" in path


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
