"""
A/B comparison visualization for segmentation methods.

This module provides functions to create side-by-side comparisons
and export high-resolution figures for publication.
"""

import matplotlib
# Use non-interactive backend to avoid Tkinter issues in tests
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from typing import Tuple, Optional, Dict, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def create_side_by_side_plot(
    rgb_image: np.ndarray,
    classic_seg: np.ndarray,
    mgrg_seg: np.ndarray,
    metrics: dict,
    title: str = "Comparativa A/B: Region Growing",
    save_path: Optional[str] = None,
    dpi: int = 300,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create side-by-side comparison plot with metrics.

    Parameters
    ----------
    rgb_image : np.ndarray
        Original RGB image (H, W, 3)
    classic_seg : np.ndarray
        Classic RG segmentation (H, W)
    mgrg_seg : np.ndarray
        MGRG segmentation (H, W)
    metrics : dict
        Dictionary with comparison metrics
    title : str
        Plot title
    save_path : Optional[str]
        Path to save figure (if None, only display)
    dpi : int
        Resolution for export (default 300 DPI)

    Returns
    -------
    Tuple[plt.Figure, np.ndarray]
        Figure object and rendered image array

    Examples
    --------
    >>> rgb = np.random.rand(100, 100, 3)
    >>> seg1 = np.random.randint(0, 5, (100, 100))
    >>> seg2 = np.random.randint(0, 3, (100, 100))
    >>> metrics = {'classic': {...}, 'mgrg': {...}}
    >>> fig, img = create_side_by_side_plot(rgb, seg1, seg2, metrics)
    """
    logger.info(f"Creating side-by-side comparison plot: {title}")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Row 1: Original + Segmentations
    axes[0, 0].imshow(rgb_image)
    axes[0, 0].set_title("Imagen Original (Sentinel-2 RGB)", fontsize=14)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(classic_seg, cmap="tab20")
    axes[0, 1].set_title(
        f'Region Growing Clásico\n'
        f'Regiones: {metrics["classic"].num_regions} | '
        f'Coherencia: {metrics["classic"].coherence:.1f}%',
        fontsize=14,
    )
    axes[0, 1].axis("off")

    axes[0, 2].imshow(mgrg_seg, cmap="tab20")
    axes[0, 2].set_title(
        f'MGRG (Semántico)\n'
        f'Regiones: {metrics["mgrg"].num_regions} | '
        f'Coherencia: {metrics["mgrg"].coherence:.1f}%',
        fontsize=14,
    )
    axes[0, 2].axis("off")

    # Row 2: Overlays + Metrics Table
    # Overlay Classic
    overlay_classic = rgb_image.copy()
    if overlay_classic.dtype != np.uint8:
        overlay_classic = (overlay_classic * 255).astype(np.uint8)
    overlay_classic[classic_seg == 0] = [255, 0, 0]  # Red for unlabeled
    axes[1, 0].imshow(overlay_classic)
    axes[1, 0].set_title("Overlay Clásico", fontsize=14)
    axes[1, 0].axis("off")

    # Overlay MGRG
    overlay_mgrg = rgb_image.copy()
    if overlay_mgrg.dtype != np.uint8:
        overlay_mgrg = (overlay_mgrg * 255).astype(np.uint8)
    overlay_mgrg[mgrg_seg == 0] = [255, 0, 0]  # Red for unlabeled
    axes[1, 1].imshow(overlay_mgrg)
    axes[1, 1].set_title("Overlay MGRG", fontsize=14)
    axes[1, 1].axis("off")

    # Metrics Table
    axes[1, 2].axis("off")
    table_data = [
        ["Métrica", "Clásico", "MGRG", "Diferencia"],
        [
            "Regiones",
            f"{metrics['classic'].num_regions}",
            f"{metrics['mgrg'].num_regions}",
            f"{metrics['differences']['num_regions']:+d}",
        ],
        [
            "Coherencia (%)",
            f"{metrics['classic'].coherence:.1f}",
            f"{metrics['mgrg'].coherence:.1f}",
            f"{metrics['differences']['coherence']:+.1f}",
        ],
        [
            "Tamaño Prom (px)",
            f"{metrics['classic'].avg_region_size:.0f}",
            f"{metrics['mgrg'].avg_region_size:.0f}",
            f"{metrics['differences']['avg_size']:+.0f}",
        ],
        [
            "Tiempo (s)",
            f"{metrics['classic'].processing_time:.2f}",
            f"{metrics['mgrg'].processing_time:.2f}",
            f"{metrics['differences']['time']:+.2f}",
        ],
    ]

    table = axes[1, 2].table(
        cellText=table_data,
        cellLoc="center",
        loc="center",
        colWidths=[0.3, 0.2, 0.2, 0.3],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)

    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor("#4CAF50")
        table[(0, i)].set_text_props(weight="bold", color="white")

    plt.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Comparison plot saved to {save_path} at {dpi} DPI")

    # Get rendered image (compatible with matplotlib >= 3.5)
    fig.canvas.draw()
    # Use buffer_rgba() instead of deprecated tostring_rgb()
    buf = fig.canvas.buffer_rgba()
    img_array = np.asarray(buf)
    # Convert RGBA to RGB by dropping alpha channel
    img_array = img_array[:, :, :3].copy()

    return fig, img_array


def create_metrics_table(
    metrics: Dict[str, any], save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create standalone metrics comparison table.

    Parameters
    ----------
    metrics : Dict[str, any]
        Dictionary with comparison metrics
    save_path : Optional[str]
        Path to save figure

    Returns
    -------
    plt.Figure
        Figure object with metrics table

    Examples
    --------
    >>> metrics = {'classic': {...}, 'mgrg': {...}, 'differences': {...}}
    >>> fig = create_metrics_table(metrics)
    """
    logger.info("Creating standalone metrics table")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")

    table_data = [
        ["Métrica", "Classic RG", "MGRG", "Diferencia", "Mejora (%)"],
        [
            "Número de Regiones",
            f"{metrics['classic'].num_regions}",
            f"{metrics['mgrg'].num_regions}",
            f"{metrics['differences']['num_regions']:+d}",
            f"{(metrics['differences']['num_regions'] / metrics['classic'].num_regions * 100):+.1f}",
        ],
        [
            "Coherencia Espacial (%)",
            f"{metrics['classic'].coherence:.2f}",
            f"{metrics['mgrg'].coherence:.2f}",
            f"{metrics['differences']['coherence']:+.2f}",
            f"{(metrics['differences']['coherence'] / metrics['classic'].coherence * 100):+.1f}",
        ],
        [
            "Tamaño Promedio (px)",
            f"{metrics['classic'].avg_region_size:.0f}",
            f"{metrics['mgrg'].avg_region_size:.0f}",
            f"{metrics['differences']['avg_size']:+.0f}",
            f"{(metrics['differences']['avg_size'] / metrics['classic'].avg_region_size * 100):+.1f}",
        ],
        [
            "Tiempo de Procesamiento (s)",
            f"{metrics['classic'].processing_time:.3f}",
            f"{metrics['mgrg'].processing_time:.3f}",
            f"{metrics['differences']['time']:+.3f}",
            f"{(metrics['differences']['time'] / metrics['classic'].processing_time * 100):+.1f}",
        ],
        [
            "Método Ganador",
            "",
            "",
            metrics["winner"].upper(),
            "",
        ],
    ]

    table = ax.table(
        cellText=table_data, cellLoc="center", loc="center", colWidths=[0.3, 0.15, 0.15, 0.2, 0.2]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2.5)

    # Style header row
    for i in range(5):
        table[(0, i)].set_facecolor("#2196F3")
        table[(0, i)].set_text_props(weight="bold", color="white")

    # Highlight winner row
    for i in range(5):
        table[(5, i)].set_facecolor("#FFC107")
        table[(5, i)].set_text_props(weight="bold")

    fig.suptitle(
        "Tabla Comparativa: Region Growing Clásico vs MGRG",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Metrics table saved to {save_path}")

    return fig


def create_overlay_comparison(
    rgb_image: np.ndarray,
    classic_seg: np.ndarray,
    mgrg_seg: np.ndarray,
    alpha: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create overlay images with segmentation boundaries.

    Parameters
    ----------
    rgb_image : np.ndarray
        Original RGB image
    classic_seg : np.ndarray
        Classic RG segmentation
    mgrg_seg : np.ndarray
        MGRG segmentation
    alpha : float
        Transparency for overlay (0-1)

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Overlay images for classic and MGRG

    Examples
    --------
    >>> rgb = np.random.rand(100, 100, 3)
    >>> seg1 = np.random.randint(0, 5, (100, 100))
    >>> seg2 = np.random.randint(0, 3, (100, 100))
    >>> overlay_classic, overlay_mgrg = create_overlay_comparison(rgb, seg1, seg2)
    """
    logger.info("Creating overlay comparisons")

    # Normalize RGB if needed
    if rgb_image.max() <= 1.0:
        rgb_normalized = (rgb_image * 255).astype(np.uint8)
    else:
        rgb_normalized = rgb_image.astype(np.uint8)

    # Create classic overlay
    overlay_classic = rgb_normalized.copy()
    classic_colored = plt.cm.tab20(
        classic_seg / (classic_seg.max() + 1e-8)
    )[:, :, :3]
    classic_colored = (classic_colored * 255).astype(np.uint8)
    overlay_classic = (
        alpha * classic_colored + (1 - alpha) * overlay_classic
    ).astype(np.uint8)

    # Create MGRG overlay
    overlay_mgrg = rgb_normalized.copy()
    mgrg_colored = plt.cm.tab20(mgrg_seg / (mgrg_seg.max() + 1e-8))[:, :, :3]
    mgrg_colored = (mgrg_colored * 255).astype(np.uint8)
    overlay_mgrg = (alpha * mgrg_colored + (1 - alpha) * overlay_mgrg).astype(
        np.uint8
    )

    logger.info("Overlay comparisons created successfully")

    return overlay_classic, overlay_mgrg


def export_high_resolution(
    fig: plt.Figure,
    save_path: str,
    dpi: int = 300,
    formats: list = ["png", "pdf", "svg"],
) -> Dict[str, str]:
    """
    Export figure in multiple high-resolution formats.

    Parameters
    ----------
    fig : plt.Figure
        Figure to export
    save_path : str
        Base path for export (without extension)
    dpi : int
        Resolution for raster formats
    formats : list
        List of formats to export

    Returns
    -------
    Dict[str, str]
        Dictionary mapping format to file path

    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3], [1, 2, 3])
    >>> exported = export_high_resolution(fig, 'output/comparison', dpi=300)
    >>> print(exported['png'])
    'output/comparison.png'
    """
    logger.info(f"Exporting figure to {len(formats)} formats at {dpi} DPI")

    exported_files = {}

    # Ensure parent directory exists
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        file_path = f"{save_path}.{fmt}"
        if fmt in ["png", "jpg"]:
            fig.savefig(file_path, dpi=dpi, bbox_inches="tight", format=fmt)
        else:
            fig.savefig(file_path, bbox_inches="tight", format=fmt)

        exported_files[fmt] = file_path
        logger.info(f"Exported {fmt.upper()} to {file_path}")

    return exported_files


def generate_failure_case_analysis(
    zone_name: str,
    rgb_image: np.ndarray,
    classic_seg: np.ndarray,
    mgrg_seg: np.ndarray,
    ndvi: np.ndarray,
    failure_description: str,
    save_dir: str,
    return_fig: bool = False,
) -> Union[str, Tuple[str, plt.Figure]]:
    """
    Generate comprehensive failure case analysis with multiple views.

    Parameters
    ----------
    zone_name : str
        Name of the zone (e.g., 'mexicali_cloud_shadow')
    rgb_image : np.ndarray
        Original RGB image
    classic_seg : np.ndarray
        Classic RG segmentation
    mgrg_seg : np.ndarray
        MGRG segmentation
    ndvi : np.ndarray
        NDVI values
    failure_description : str
        Description of the failure case
    save_dir : str
        Directory to save analysis
    return_fig : bool, optional
        If True, return figure object for display in notebooks (default: False)

    Returns
    -------
    str or Tuple[str, plt.Figure]
        Path to saved analysis figure, or tuple of (path, figure) if return_fig=True

    Examples
    --------
    >>> zone = 'test_zone'
    >>> rgb = np.random.rand(100, 100, 3)
    >>> seg1 = np.random.randint(0, 5, (100, 100))
    >>> seg2 = np.random.randint(0, 3, (100, 100))
    >>> ndvi = np.random.rand(100, 100)
    >>> desc = 'Test failure case'
    >>> path = generate_failure_case_analysis(zone, rgb, seg1, seg2, ndvi, desc, 'output')
    """
    logger.info(f"Generating failure case analysis for {zone_name}")

    fig, axes = plt.subplots(2, 3, figsize=(20, 13))

    # Row 1: RGB, NDVI, Classic
    axes[0, 0].imshow(rgb_image)
    axes[0, 0].set_title("RGB Original", fontsize=14)
    axes[0, 0].axis("off")

    ndvi_plot = axes[0, 1].imshow(ndvi, cmap="RdYlGn", vmin=-1, vmax=1)
    axes[0, 1].set_title("NDVI", fontsize=14)
    axes[0, 1].axis("off")
    cbar1 = plt.colorbar(ndvi_plot, ax=axes[0, 1], fraction=0.046)
    cbar1.set_label("NDVI", rotation=270, labelpad=15)

    axes[0, 2].imshow(classic_seg, cmap="tab20")
    axes[0, 2].set_title("Segmentación Clásica", fontsize=14)
    axes[0, 2].axis("off")

    # Row 2: MGRG, Difference, Description
    axes[1, 0].imshow(mgrg_seg, cmap="tab20")
    axes[1, 0].set_title("Segmentación MGRG", fontsize=14)
    axes[1, 0].axis("off")

    # Difference map
    diff = (classic_seg != mgrg_seg).astype(int)
    diff_plot = axes[1, 1].imshow(diff, cmap="Reds")
    axes[1, 1].set_title("Diferencias (Rojo)", fontsize=14)
    axes[1, 1].axis("off")

    # Description text
    axes[1, 2].axis("off")
    axes[1, 2].text(
        0.1,
        0.9,
        f"Caso de Fallo: {zone_name}\n\n{failure_description}",
        fontsize=12,
        verticalalignment="top",
        wrap=True,
    )

    plt.suptitle(
        f"Análisis de Caso de Fallo: {zone_name}",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()

    # Ensure directory exists
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    save_path = f"{save_dir}/{zone_name}_failure_analysis.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    logger.info(f"Failure case analysis saved to {save_path}")

    if return_fig:
        return save_path, fig
    else:
        plt.close(fig)
        return save_path
