"""
Visualization utilities for satellite imagery analysis.

This module provides functions for creating multi-spectral visualizations
including RGB composites, false color, and NDVI maps.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional, Tuple


def create_rgb_composite(hls_image: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Create RGB natural color composite from HLS image.
    
    Parameters
    ----------
    hls_image : np.ndarray
        HLS image with shape (6, H, W) containing bands in order:
        [B02, B03, B04, B8A, B11, B12]
    normalize : bool, default=True
        Whether to normalize values to [0, 1] range
        
    Returns
    -------
    np.ndarray
        RGB composite with shape (H, W, 3)
        
    Examples
    --------
    >>> hls_image = np.random.rand(6, 512, 512)
    >>> rgb = create_rgb_composite(hls_image)
    >>> print(rgb.shape)
    (512, 512, 3)
    """
    # Stack B04 (Red), B03 (Green), B02 (Blue)
    rgb = np.stack([
        hls_image[2],  # B04 (Red)
        hls_image[1],  # B03 (Green)
        hls_image[0]   # B02 (Blue)
    ], axis=-1)
    
    if normalize:
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
    
    return rgb


def create_false_color_composite(hls_image: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Create false color composite (NIR-Red-Green) from HLS image.
    
    Useful for vegetation analysis as healthy vegetation appears bright red.
    
    Parameters
    ----------
    hls_image : np.ndarray
        HLS image with shape (6, H, W) containing bands in order:
        [B02, B03, B04, B8A, B11, B12]
    normalize : bool, default=True
        Whether to normalize values to [0, 1] range
        
    Returns
    -------
    np.ndarray
        False color composite with shape (H, W, 3)
        
    Examples
    --------
    >>> hls_image = np.random.rand(6, 512, 512)
    >>> false_color = create_false_color_composite(hls_image)
    >>> print(false_color.shape)
    (512, 512, 3)
    """
    # Stack B8A (NIR), B04 (Red), B03 (Green)
    false_color = np.stack([
        hls_image[3],  # B8A (NIR)
        hls_image[2],  # B04 (Red)
        hls_image[1]   # B03 (Green)
    ], axis=-1)
    
    if normalize:
        false_color = (false_color - false_color.min()) / (false_color.max() - false_color.min() + 1e-8)
    
    return false_color


def compute_ndvi_from_hls(hls_image: np.ndarray) -> np.ndarray:
    """
    Compute NDVI (Normalized Difference Vegetation Index) from HLS image.
    
    NDVI = (NIR - Red) / (NIR + Red)
    
    Parameters
    ----------
    hls_image : np.ndarray
        HLS image with shape (6, H, W) containing bands in order:
        [B02, B03, B04, B8A, B11, B12]
        
    Returns
    -------
    np.ndarray
        NDVI values with shape (H, W), range [-1, 1]
        
    Examples
    --------
    >>> hls_image = np.random.rand(6, 512, 512)
    >>> ndvi = compute_ndvi_from_hls(hls_image)
    >>> print(ndvi.shape)
    (512, 512)
    """
    nir = hls_image[3]  # B8A (NIR)
    red = hls_image[2]  # B04 (Red)
    
    ndvi = (nir - red) / (nir + red + 1e-8)
    
    return ndvi


def visualize_multispectral_zones(
    zones_data: Dict,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (18, 16),
    dpi: int = 150,
    show_plot: bool = True
) -> plt.Figure:
    """
    Create multi-spectral visualization grid for multiple zones.
    
    Creates a 3xN grid showing RGB, False Color, and NDVI for each zone.
    
    Parameters
    ----------
    zones_data : dict
        Dictionary with zone data from load_or_download_zones()
        Format: {zone_id: {'config': {...}, 'hls_image': array, ...}}
    output_path : Path, optional
        Path to save the figure. If None, figure is not saved.
    figsize : tuple, default=(18, 16)
        Figure size in inches (width, height)
    dpi : int, default=150
        Resolution for saved figure
    show_plot : bool, default=True
        Whether to display the plot
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure object
        
    Examples
    --------
    >>> from pathlib import Path
    >>> zones_data = load_or_download_zones(...)
    >>> fig = visualize_multispectral_zones(
    ...     zones_data,
    ...     output_path=Path('output/multispectral.png')
    ... )
    """
    # Filter out None zones
    valid_zones = {k: v for k, v in zones_data.items() if v is not None}
    n_zones = len(valid_zones)
    
    if n_zones == 0:
        raise ValueError("No valid zones to visualize")
    
    # Create figure
    fig, axes = plt.subplots(3, n_zones, figsize=figsize)
    
    # Handle single zone case (axes won't be 2D)
    if n_zones == 1:
        axes = axes.reshape(3, 1)
    
    for idx, (zone_id, zone_data) in enumerate(valid_zones.items()):
        hls_image = zone_data['hls_image']
        zone_name = zone_data['config']['name']
        
        # 1. RGB Natural Color
        rgb = create_rgb_composite(hls_image)
        axes[0, idx].imshow(rgb)
        axes[0, idx].set_title(
            f"{zone_name}\nRGB Natural",
            fontsize=10,
            fontweight='bold'
        )
        axes[0, idx].axis('off')
        
        # 2. False Color (NIR-Red-Green)
        false_color = create_false_color_composite(hls_image)
        axes[1, idx].imshow(false_color)
        axes[1, idx].set_title(
            "Falso Color (NIR-R-G)",
            fontsize=10,
            fontweight='bold'
        )
        axes[1, idx].axis('off')
        
        # 3. NDVI
        ndvi = compute_ndvi_from_hls(hls_image)
        im = axes[2, idx].imshow(ndvi, cmap='RdYlGn', vmin=-0.2, vmax=0.8)
        axes[2, idx].set_title("NDVI", fontsize=10, fontweight='bold')
        axes[2, idx].axis('off')
        
        # Add colorbar for NDVI
        plt.colorbar(im, ax=axes[2, idx], fraction=0.046, pad=0.04)
    
    # Add row labels
    axes[0, 0].text(
        -0.1, 0.5, 'RGB Natural',
        transform=axes[0, 0].transAxes,
        fontsize=12, fontweight='bold',
        va='center', rotation=90
    )
    axes[1, 0].text(
        -0.1, 0.5, 'Falso Color',
        transform=axes[1, 0].transAxes,
        fontsize=12, fontweight='bold',
        va='center', rotation=90
    )
    axes[2, 0].text(
        -0.1, 0.5, 'NDVI',
        transform=axes[2, 0].transAxes,
        fontsize=12, fontweight='bold',
        va='center', rotation=90
    )
    
    # Add title
    plt.suptitle(
        'Visualización Multi-espectral: RGB, Falso Color y NDVI',
        fontsize=14,
        fontweight='bold',
        y=0.995
    )
    
    plt.tight_layout()
    
    # Save if output path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Visualización guardada en: {output_path}")
    
    # Show plot
    if show_plot:
        plt.show()
    
    return fig


def visualize_single_zone(
    hls_image: np.ndarray,
    zone_name: str,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (15, 5),
    dpi: int = 150,
    show_plot: bool = True
) -> plt.Figure:
    """
    Create multi-spectral visualization for a single zone.
    
    Creates a 1x3 plot showing RGB, False Color, and NDVI.
    
    Parameters
    ----------
    hls_image : np.ndarray
        HLS image with shape (6, H, W)
    zone_name : str
        Name of the zone for title
    output_path : Path, optional
        Path to save the figure
    figsize : tuple, default=(15, 5)
        Figure size in inches (width, height)
    dpi : int, default=150
        Resolution for saved figure
    show_plot : bool, default=True
        Whether to display the plot
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure object
        
    Examples
    --------
    >>> hls_image = np.random.rand(6, 512, 512)
    >>> fig = visualize_single_zone(
    ...     hls_image,
    ...     zone_name='Mexicali',
    ...     output_path=Path('output/mexicali.png')
    ... )
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # 1. RGB Natural Color
    rgb = create_rgb_composite(hls_image)
    axes[0].imshow(rgb)
    axes[0].set_title('RGB Natural', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # 2. False Color
    false_color = create_false_color_composite(hls_image)
    axes[1].imshow(false_color)
    axes[1].set_title('Falso Color (NIR-R-G)', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # 3. NDVI
    ndvi = compute_ndvi_from_hls(hls_image)
    im = axes[2].imshow(ndvi, cmap='RdYlGn', vmin=-0.2, vmax=0.8)
    axes[2].set_title('NDVI', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.suptitle(
        f'Visualización Multi-espectral: {zone_name}',
        fontsize=14,
        fontweight='bold'
    )
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Visualización guardada en: {output_path}")
    
    if show_plot:
        plt.show()
    
    return fig



def visualize_segmentation_results(
    zones_data: Dict,
    ndvi_results: Dict,
    segmentation_results: Dict,
    stress_classification: Dict,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (18, 16),
    dpi: int = 300,
    show_plot: bool = True
) -> plt.Figure:
    """
    Visualize complete segmentation and classification results for multiple zones.
    
    Creates a 3xN grid showing NDVI, Segmentation, and Stress Classification
    for each zone. Works with both Classic RG and MGRG results.
    
    Parameters
    ----------
    zones_data : dict
        Dictionary with zone configuration data
    ndvi_results : dict
        Dictionary with NDVI arrays for each zone
        Format: {zone_id: ndvi_array}
    segmentation_results : dict
        Dictionary with segmentation results for each zone
        Format: {zone_id: {'labeled': array, 'num_regions': int}}
    stress_classification : dict
        Dictionary with stress classification results for each zone
        Format: {zone_id: {'classified': {'high_stress': [...], 'medium_stress': [...], 'low_stress': [...]}}}
    output_path : Path, optional
        Path to save the figure
    figsize : tuple, default=(18, 16)
        Figure size in inches (width, height)
    dpi : int, default=300
        Resolution for saved figure
    show_plot : bool, default=True
        Whether to display the plot
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure object
        
    Examples
    --------
    >>> # For Classic RG
    >>> fig = visualize_segmentation_results(
    ...     zones_data=zones_data,
    ...     ndvi_results=ndvi_results,
    ...     segmentation_results=segmentation_results,
    ...     stress_classification=classic_stress_classification,
    ...     output_path=Path('output/classic_rg_results.png')
    ... )
    >>> 
    >>> # For MGRG
    >>> fig = visualize_segmentation_results(
    ...     zones_data=zones_data,
    ...     ndvi_results=ndvi_results,
    ...     segmentation_results=mgrg_results,
    ...     stress_classification=mgrg_stress_classification,
    ...     output_path=Path('output/mgrg_results.png')
    ... )
    """
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    
    # Get valid zones
    valid_zones = [zid for zid in zones_data.keys() if zid in segmentation_results]
    n_zones = len(valid_zones)
    
    if n_zones == 0:
        raise ValueError("No valid zones with segmentation results to visualize")
    
    # Create figure
    fig, axes = plt.subplots(3, n_zones, figsize=figsize)
    
    # Handle single zone case
    if n_zones == 1:
        axes = axes.reshape(3, 1)
    
    for idx, zone_id in enumerate(valid_zones):
        zone_name = zones_data[zone_id]['config']['name']
        ndvi = ndvi_results[zone_id]
        labeled = segmentation_results[zone_id]['labeled']
        num_regions = segmentation_results[zone_id]['num_regions']
        classified = stress_classification[zone_id]['classified']
        
        # Row 1: NDVI
        im0 = axes[0, idx].imshow(ndvi, cmap='RdYlGn', vmin=-0.2, vmax=0.8)
        axes[0, idx].set_title(
            f"{zone_name}\nNDVI",
            fontsize=10,
            fontweight='bold'
        )
        axes[0, idx].axis('off')
        plt.colorbar(im0, ax=axes[0, idx], fraction=0.046, pad=0.04)
        
        # Row 2: Segmentation
        im1 = axes[1, idx].imshow(labeled, cmap='tab20')
        axes[1, idx].set_title(
            f"Segmentación\n({num_regions} regiones)",
            fontsize=10,
            fontweight='bold'
        )
        axes[1, idx].axis('off')
        
        # Row 3: Stress Map
        stress_map = np.zeros_like(labeled, dtype=np.uint8)
        
        # Iterate over each stress category
        for region in classified['high_stress']:
            mask = (labeled == region['id'])
            stress_map[mask] = 3
        
        for region in classified['medium_stress']:
            mask = (labeled == region['id'])
            stress_map[mask] = 2
        
        for region in classified['low_stress']:
            mask = (labeled == region['id'])
            stress_map[mask] = 1
        
        colors = ['black', 'green', 'yellow', 'red']
        cmap_stress = ListedColormap(colors)
        im2 = axes[2, idx].imshow(stress_map, cmap=cmap_stress, vmin=0, vmax=3)
        axes[2, idx].set_title(
            "Estrés Vegetal",
            fontsize=10,
            fontweight='bold'
        )
        axes[2, idx].axis('off')
    
    # Add row labels
    axes[0, 0].text(
        -0.1, 0.5, 'NDVI',
        transform=axes[0, 0].transAxes,
        fontsize=12, fontweight='bold',
        va='center', rotation=90
    )
    axes[1, 0].text(
        -0.1, 0.5, 'Segmentación',
        transform=axes[1, 0].transAxes,
        fontsize=12, fontweight='bold',
        va='center', rotation=90
    )
    axes[2, 0].text(
        -0.1, 0.5, 'Clasificación',
        transform=axes[2, 0].transAxes,
        fontsize=12, fontweight='bold',
        va='center', rotation=90
    )
    
    # Add legend
    legend_elements = [
        Patch(facecolor='green', label='Bajo (NDVI ≥ 0.5)'),
        Patch(facecolor='yellow', label='Medio (0.3 ≤ NDVI < 0.5)'),
        Patch(facecolor='red', label='Alto (NDVI < 0.3)')
    ]
    axes[2, n_zones-1].legend(
        handles=legend_elements,
        loc='center left',
        bbox_to_anchor=(1.05, 0.5),
        fontsize=9
    )
    
    plt.suptitle(
        'Classic Region Growing: Segmentación y Clasificación de Estrés Vegetal',
        fontsize=14,
        fontweight='bold',
        y=0.995
    )
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Visualización completa guardada en: {output_path}")
    
    if show_plot:
        plt.show()
    
    return fig



def visualize_method_comparison(
    zones_data: Dict,
    ndvi_results: Dict,
    classic_results: Dict,
    classic_stress: Dict,
    mgrg_results: Dict,
    mgrg_stress: Dict,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (18, 20),
    dpi: int = 150,
    show_plot: bool = True
) -> plt.Figure:
    """
    Create comprehensive comparison visualization between Classic RG and MGRG.
    
    Creates a 4xN grid showing:
    - Row 1: NDVI
    - Row 2: Classic RG Segmentation
    - Row 3: MGRG Segmentation
    - Row 4: Side-by-side stress comparison
    
    Parameters
    ----------
    zones_data : dict
        Dictionary with zone configuration data
    ndvi_results : dict
        Dictionary with NDVI arrays for each zone
    classic_results : dict
        Classic RG segmentation results
    classic_stress : dict
        Classic RG stress classification
    mgrg_results : dict
        MGRG segmentation results
    mgrg_stress : dict
        MGRG stress classification
    output_path : Path, optional
        Path to save the figure
    figsize : tuple, default=(18, 20)
        Figure size in inches (width, height)
    dpi : int, default=150
        Resolution for saved figure
    show_plot : bool, default=True
        Whether to display the plot
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure object
        
    Examples
    --------
    >>> fig = visualize_method_comparison(
    ...     zones_data=zones_data,
    ...     ndvi_results=ndvi_results,
    ...     classic_results=segmentation_results,
    ...     classic_stress=classic_stress_classification,
    ...     mgrg_results=mgrg_results,
    ...     mgrg_stress=mgrg_stress_classification,
    ...     output_path=Path('output/comparison.png')
    ... )
    """
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    
    # Get valid zones
    valid_zones = [
        zid for zid in zones_data.keys() 
        if zid in classic_results and zid in mgrg_results
    ]
    n_zones = len(valid_zones)
    
    if n_zones == 0:
        raise ValueError("No valid zones with both Classic RG and MGRG results")
    
    # Create figure
    fig, axes = plt.subplots(4, n_zones, figsize=figsize)
    
    # Handle single zone case
    if n_zones == 1:
        axes = axes.reshape(4, 1)
    
    for idx, zone_id in enumerate(valid_zones):
        zone_name = zones_data[zone_id]['config']['name']
        
        # Get data
        ndvi = ndvi_results[zone_id]
        classic_labeled = classic_results[zone_id]['labeled']
        classic_num = classic_results[zone_id]['num_regions']
        classic_classified = classic_stress[zone_id]['classified']
        mgrg_labeled = mgrg_results[zone_id]['labeled']
        mgrg_num = mgrg_results[zone_id]['num_regions']
        mgrg_classified = mgrg_stress[zone_id]['classified']
        
        # Row 1: NDVI
        im0 = axes[0, idx].imshow(ndvi, cmap='RdYlGn', vmin=-0.2, vmax=0.8)
        axes[0, idx].set_title(
            f"{zone_name}\nNDVI",
            fontsize=10,
            fontweight='bold'
        )
        axes[0, idx].axis('off')
        plt.colorbar(im0, ax=axes[0, idx], fraction=0.046, pad=0.04)
        
        # Row 2: Classic RG Segmentation
        im1 = axes[1, idx].imshow(classic_labeled, cmap='tab20')
        axes[1, idx].set_title(
            f"Classic RG\n({classic_num} regiones)",
            fontsize=10,
            fontweight='bold'
        )
        axes[1, idx].axis('off')
        
        # Row 3: MGRG Segmentation
        im2 = axes[2, idx].imshow(mgrg_labeled, cmap='tab20')
        axes[2, idx].set_title(
            f"MGRG\n({mgrg_num} regiones)",
            fontsize=10,
            fontweight='bold'
        )
        axes[2, idx].axis('off')
        
        # Row 4: Stress Comparison
        # Classic stress map
        stress_map_classic = np.zeros_like(classic_labeled, dtype=np.uint8)
        for region in classic_classified['high_stress']:
            mask = (classic_labeled == region['id'])
            stress_map_classic[mask] = 3
        for region in classic_classified['medium_stress']:
            mask = (classic_labeled == region['id'])
            stress_map_classic[mask] = 2
        for region in classic_classified['low_stress']:
            mask = (classic_labeled == region['id'])
            stress_map_classic[mask] = 1
        
        # MGRG stress map
        stress_map_mgrg = np.zeros_like(mgrg_labeled, dtype=np.uint8)
        for region in mgrg_classified['high_stress']:
            mask = (mgrg_labeled == region['id'])
            stress_map_mgrg[mask] = 3
        for region in mgrg_classified['medium_stress']:
            mask = (mgrg_labeled == region['id'])
            stress_map_mgrg[mask] = 2
        for region in mgrg_classified['low_stress']:
            mask = (mgrg_labeled == region['id'])
            stress_map_mgrg[mask] = 1
        
        # Create combined view (side by side)
        combined_stress = np.hstack([stress_map_classic, stress_map_mgrg])
        
        colors = ['black', 'green', 'yellow', 'red']
        cmap_stress = ListedColormap(colors)
        im3 = axes[3, idx].imshow(combined_stress, cmap=cmap_stress, vmin=0, vmax=3)
        axes[3, idx].set_title(
            "Estrés: Classic (izq) vs MGRG (der)",
            fontsize=9,
            fontweight='bold'
        )
        axes[3, idx].axis('off')
    
    # Add row labels
    axes[0, 0].text(
        -0.1, 0.5, 'NDVI',
        transform=axes[0, 0].transAxes,
        fontsize=12, fontweight='bold',
        va='center', rotation=90
    )
    axes[1, 0].text(
        -0.1, 0.5, 'Classic RG',
        transform=axes[1, 0].transAxes,
        fontsize=12, fontweight='bold',
        va='center', rotation=90
    )
    axes[2, 0].text(
        -0.1, 0.5, 'MGRG',
        transform=axes[2, 0].transAxes,
        fontsize=12, fontweight='bold',
        va='center', rotation=90
    )
    axes[3, 0].text(
        -0.1, 0.5, 'Comparación',
        transform=axes[3, 0].transAxes,
        fontsize=12, fontweight='bold',
        va='center', rotation=90
    )
    
    # Add legend
    legend_elements = [
        Patch(facecolor='green', label='Bajo (NDVI ≥ 0.5)'),
        Patch(facecolor='yellow', label='Medio (0.3 ≤ NDVI < 0.5)'),
        Patch(facecolor='red', label='Alto (NDVI < 0.3)')
    ]
    axes[3, n_zones-1].legend(
        handles=legend_elements,
        loc='center left',
        bbox_to_anchor=(1.05, 0.5),
        fontsize=9
    )
    
    plt.suptitle(
        'Comparación Completa: Classic Region Growing vs MGRG',
        fontsize=14,
        fontweight='bold',
        y=0.995
    )
    
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Visualización comparativa guardada en: {output_path}")
    
    if show_plot:
        plt.show()
    
    return fig



def visualize_ab_comparison_metrics(
    zones_data: Dict,
    classic_results: Dict,
    classic_quality: Dict,
    mgrg_results: Dict,
    mgrg_quality: Dict,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (16, 20),
    dpi: int = 300,
    show_plot: bool = True
):
    """
    Create A/B comparison charts for Classic RG vs MGRG metrics.
    
    Creates a 3x2 grid of bar charts comparing:
    - Number of regions
    - Spatial coherence
    - Average region size
    - Processing time
    - Region reduction percentage
    - Coherence improvement
    
    Parameters
    ----------
    zones_data : dict
        Dictionary with zone configuration data
    classic_results : dict
        Classic RG segmentation results
    classic_quality : dict
        Classic RG quality metrics
    mgrg_results : dict
        MGRG segmentation results
    mgrg_quality : dict
        MGRG quality metrics
    output_path : Path, optional
        Path to save the figure
    figsize : tuple, default=(16, 20)
        Figure size in inches (width, height)
    dpi : int, default=300
        Resolution for saved figure
    show_plot : bool, default=True
        Whether to display the plot
        
    Returns
    -------
    tuple
        (fig, summary_dict) where:
        - fig: matplotlib Figure object
        - summary_dict: Dictionary with average improvements
        
    Examples
    --------
    >>> fig, summary = visualize_ab_comparison_metrics(
    ...     zones_data=zones_data,
    ...     classic_results=segmentation_results,
    ...     classic_quality=classic_quality_metrics,
    ...     mgrg_results=mgrg_results,
    ...     mgrg_quality=mgrg_quality_metrics,
    ...     output_path=Path('output/ab_comparison.png')
    ... )
    >>> print(f"Reducción promedio: {summary['avg_reduction']:.1f}%")
    """
    # Prepare data
    zones_list = [zid for zid in zones_data.keys() if zid in classic_results and zid in mgrg_results]
    zones_labels = [zones_data[zid]['config']['name'] for zid in zones_list]
    x = np.arange(len(zones_list))
    width = 0.4
    
    # Create figure
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    
    # 1. Number of regions
    classic_regions = [classic_results[zid]['num_regions'] for zid in zones_list]
    mgrg_regions = [mgrg_results[zid]['num_regions'] for zid in zones_list]
    
    axes[0, 0].bar(x - width/2, classic_regions, width, label='Classic RG', alpha=0.8, color='steelblue')
    axes[0, 0].bar(x + width/2, mgrg_regions, width, label='MGRG', alpha=0.8, color='coral')
    axes[0, 0].set_ylabel('Número de Regiones', fontweight='bold', fontsize=11)
    axes[0, 0].set_title('Número de Regiones por Método', fontweight='bold', fontsize=12)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(zones_labels, fontsize=10)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Add values on bars
    for i, (c, m) in enumerate(zip(classic_regions, mgrg_regions)):
        axes[0, 0].text(i - width/2, c + max(classic_regions)*0.02, str(c), 
                        ha='center', va='bottom', fontsize=9)
        axes[0, 0].text(i + width/2, m + max(classic_regions)*0.02, str(m), 
                        ha='center', va='bottom', fontsize=9)
    
    # 2. Spatial coherence
    classic_coherence = [classic_quality[zid]['coherence'] for zid in zones_list]
    mgrg_coherence = [mgrg_quality[zid]['coherence'] for zid in zones_list]
    
    axes[0, 1].bar(x - width/2, classic_coherence, width, label='Classic RG', alpha=0.8, color='steelblue')
    axes[0, 1].bar(x + width/2, mgrg_coherence, width, label='MGRG', alpha=0.8, color='coral')
    axes[0, 1].set_ylabel('Coherencia (%)', fontweight='bold', fontsize=11)
    axes[0, 1].set_title('Coherencia Espacial por Método', fontweight='bold', fontsize=12)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(zones_labels, fontsize=10)
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(axis='y', alpha=0.3)
    axes[0, 1].set_ylim(0, 100)
    
    # Add values
    for i, (c, m) in enumerate(zip(classic_coherence, mgrg_coherence)):
        axes[0, 1].text(i - width/2, c + 2, f'{c:.1f}', ha='center', va='bottom', fontsize=9)
        axes[0, 1].text(i + width/2, m + 2, f'{m:.1f}', ha='center', va='bottom', fontsize=9)
    
    # 3. Average region size
    classic_size = [classic_quality[zid]['avg_size'] for zid in zones_list]
    mgrg_size = [mgrg_quality[zid]['avg_size'] for zid in zones_list]
    
    axes[1, 0].bar(x - width/2, classic_size, width, label='Classic RG', alpha=0.8, color='steelblue')
    axes[1, 0].bar(x + width/2, mgrg_size, width, label='MGRG', alpha=0.8, color='coral')
    axes[1, 0].set_ylabel('Tamaño Promedio (píxeles)', fontweight='bold', fontsize=11)
    axes[1, 0].set_title('Tamaño Promedio de Región', fontweight='bold', fontsize=12)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(zones_labels, fontsize=10)
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # 4. Processing time
    classic_time = [classic_results[zid].get('elapsed_time', 0) for zid in zones_list]
    mgrg_time = [mgrg_results[zid].get('elapsed_time', 0) for zid in zones_list]
    
    axes[1, 1].bar(x - width/2, classic_time, width, label='Classic RG', alpha=0.8, color='steelblue')
    axes[1, 1].bar(x + width/2, mgrg_time, width, label='MGRG', alpha=0.8, color='coral')
    axes[1, 1].set_ylabel('Tiempo (segundos)', fontweight='bold', fontsize=11)
    axes[1, 1].set_title('Tiempo de Procesamiento', fontweight='bold', fontsize=12)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(zones_labels, fontsize=10)
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    # Add values
    max_time = max(max(classic_time), max(mgrg_time))
    for i, (c, m) in enumerate(zip(classic_time, mgrg_time)):
        axes[1, 1].text(i - width/2, c + max_time*0.02, f'{c:.2f}', 
                        ha='center', va='bottom', fontsize=9)
        axes[1, 1].text(i + width/2, m + max_time*0.02, f'{m:.2f}', 
                        ha='center', va='bottom', fontsize=9)
    
    # 5. Region reduction (%)
    reduction = [
        (1 - mgrg_results[zid]['num_regions'] / classic_results[zid]['num_regions']) * 100 
        for zid in zones_list
    ]
    
    axes[2, 0].bar(x, reduction, width*2, alpha=0.8, color='green')
    axes[2, 0].set_ylabel('Reducción (%)', fontweight='bold', fontsize=11)
    axes[2, 0].set_title('Reducción de Regiones (MGRG vs Classic)', fontweight='bold', fontsize=12)
    axes[2, 0].set_xticks(x)
    axes[2, 0].set_xticklabels(zones_labels, fontsize=10)
    axes[2, 0].grid(axis='y', alpha=0.3)
    axes[2, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Add values
    for i, r in enumerate(reduction):
        axes[2, 0].text(i, r + 2, f'{r:.1f}%', ha='center', va='bottom', 
                        fontsize=10, fontweight='bold')
    
    # 6. Coherence improvement (percentage points)
    coherence_improvement = [
        mgrg_quality[zid]['coherence'] - classic_quality[zid]['coherence'] 
        for zid in zones_list
    ]
    colors = ['green' if x > 0 else 'red' for x in coherence_improvement]
    
    axes[2, 1].bar(x, coherence_improvement, width*2, alpha=0.8, color=colors)
    axes[2, 1].set_ylabel('Mejora (puntos %)', fontweight='bold', fontsize=11)
    axes[2, 1].set_title('Mejora en Coherencia (MGRG vs Classic)', fontweight='bold', fontsize=12)
    axes[2, 1].set_xticks(x)
    axes[2, 1].set_xticklabels(zones_labels, fontsize=10)
    axes[2, 1].grid(axis='y', alpha=0.3)
    axes[2, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Add values
    for i, imp in enumerate(coherence_improvement):
        axes[2, 1].text(i, imp + 0.5 if imp > 0 else imp - 0.5, f'{imp:+.1f}', 
                        ha='center', va='bottom' if imp > 0 else 'top', 
                        fontsize=10, fontweight='bold')
    
    plt.suptitle('Comparativa A/B: Classic Region Growing vs MGRG', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Gráficos de comparativa A/B guardados en: {output_path}")
    
    if show_plot:
        plt.show()
    
    # Calculate summary statistics
    avg_classic_time = np.mean(classic_time) if classic_time else 0
    avg_mgrg_time = np.mean(mgrg_time) if mgrg_time else 0
    time_factor = avg_mgrg_time / avg_classic_time if avg_classic_time > 0 else 0
    
    summary = {
        'avg_reduction': np.mean(reduction),
        'avg_coherence_improvement': np.mean(coherence_improvement),
        'avg_classic_time': avg_classic_time,
        'avg_mgrg_time': avg_mgrg_time,
        'time_factor': time_factor
    }
    
    # Print summary
    print("\n" + "="*80)
    print("RESUMEN DE MEJORAS PROMEDIO:")
    print("="*80)
    print(f"  • Reducción de regiones:      {summary['avg_reduction']:.1f}%")
    print(f"  • Mejora en coherencia:       {summary['avg_coherence_improvement']:+.1f} puntos porcentuales")
    print(f"  • Tiempo Classic RG:          {summary['avg_classic_time']:.2f}s")
    print(f"  • Tiempo MGRG:                {summary['avg_mgrg_time']:.2f}s")
    print(f"  • Factor de tiempo:           {summary['time_factor']:.1f}x")
    print("="*80 + "\n")
    
    return fig, summary



def print_ndvi_statistics(
    ndvi_results: Dict,
    zones_data: Dict,
    zone_mapping: Dict
) -> None:
    """
    Print NDVI statistics for all zones.
    
    Parameters
    ----------
    ndvi_results : dict
        Dictionary with NDVI arrays for each zone
    zones_data : dict
        Zone configuration data
    zone_mapping : dict
        Mapping from zone keys to zone IDs
        
    Examples
    --------
    >>> print_ndvi_statistics(ndvi_results, zones_data, zone_mapping)
    """
    for zone_name, zone_id in zone_mapping.items():
        if zone_id not in ndvi_results:
            continue
        
        zone_display = zones_data[zone_id]['config']['name']
        ndvi = ndvi_results[zone_id]
        
        print(f"\n{zone_display}:")
        print(f"  Min:  {ndvi.min():.3f}")
        print(f"  Max:  {ndvi.max():.3f}")
        print(f"  Mean: {ndvi.mean():.3f}")
        print(f"  Std:  {ndvi.std():.3f}")


def visualize_ndvi_distribution(
    ndvi_results: Dict,
    zones_data: Dict,
    zone_mapping: Dict,
    output_path: Optional[Path] = None,
    figsize: Optional[Tuple[int, int]] = None,
    dpi: int = 150,
    show_plot: bool = True
) -> plt.Figure:
    """
    Visualize NDVI distribution and maps for all zones.
    
    Creates a grid with histogram and NDVI map for each zone,
    showing stress and vigorous thresholds.
    
    Parameters
    ----------
    ndvi_results : dict
        Dictionary with NDVI arrays for each zone
    zones_data : dict
        Zone configuration data
    zone_mapping : dict
        Mapping from zone keys to zone IDs
    output_path : Path, optional
        Path to save the figure
    figsize : tuple, optional
        Figure size in inches (width, height). If None, auto-calculated.
    dpi : int, default=150
        Resolution for saved figure
    show_plot : bool, default=True
        Whether to display the plot
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure object
        
    Examples
    --------
    >>> fig = visualize_ndvi_distribution(
    ...     ndvi_results, zones_data, zone_mapping,
    ...     output_path=Path('output/ndvi_distribution.png')
    ... )
    """
    # Count valid zones
    valid_zones = [(zn, zid) for zn, zid in zone_mapping.items() if zid in ndvi_results]
    n_zones = len(valid_zones)
    
    if n_zones == 0:
        raise ValueError("No valid zones with NDVI results")
    
    # Auto-calculate figsize if not provided
    if figsize is None:
        figsize = (15, 5 * n_zones)
    
    # Create figure
    fig, axes = plt.subplots(n_zones, 2, figsize=figsize)
    
    # Handle single zone case
    if n_zones == 1:
        axes = axes.reshape(1, 2)
    
    for idx, (zone_name, zone_id) in enumerate(valid_zones):
        zone_display = zones_data[zone_id]['config']['name']
        ndvi = ndvi_results[zone_id]
        
        # 1. Histogram
        axes[idx, 0].hist(
            ndvi[ndvi > -1].flatten(),
            bins=50,
            edgecolor='black',
            alpha=0.7,
            color='steelblue'
        )
        axes[idx, 0].set_xlabel('NDVI', fontweight='bold')
        axes[idx, 0].set_ylabel('Frecuencia', fontweight='bold')
        axes[idx, 0].set_title(
            f'Distribución de NDVI - {zone_display}',
            fontweight='bold'
        )
        axes[idx, 0].grid(alpha=0.3)
        axes[idx, 0].axvline(x=0.3, color='orange', linestyle='--', 
                             label='Umbral Estrés', linewidth=2)
        axes[idx, 0].axvline(x=0.6, color='green', linestyle='--', 
                             label='Umbral Vigoroso', linewidth=2)
        axes[idx, 0].legend()
        
        # 2. NDVI map
        im = axes[idx, 1].imshow(ndvi, cmap='RdYlGn', vmin=-0.2, vmax=0.8)
        axes[idx, 1].set_title(
            f'Mapa de NDVI - {zone_display}',
            fontweight='bold'
        )
        axes[idx, 1].axis('off')
        plt.colorbar(im, ax=axes[idx, 1], label='NDVI', fraction=0.046, pad=0.04)
    
    plt.suptitle(
        'Análisis de NDVI por Zona',
        fontsize=16,
        fontweight='bold'
    )
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Visualización NDVI guardada en: {output_path}")
    
    if show_plot:
        plt.show()
    
    return fig


def analyze_ndvi_zones(
    ndvi_results: Dict,
    zones_data: Dict,
    zone_mapping: Dict,
    output_path: Optional[Path] = None,
    dpi: int = 150
) -> plt.Figure:
    """
    Complete NDVI analysis: statistics + visualization.
    
    Combines print_ndvi_statistics() and visualize_ndvi_distribution()
    in a single convenient function.
    
    Parameters
    ----------
    ndvi_results : dict
        Dictionary with NDVI arrays for each zone
    zones_data : dict
        Zone configuration data
    zone_mapping : dict
        Mapping from zone keys to zone IDs
    output_path : Path, optional
        Path to save the figure
    dpi : int, default=150
        Resolution for saved figure
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure object
        
    Examples
    --------
    >>> fig = analyze_ndvi_zones(
    ...     ndvi_results, zones_data, zone_mapping,
    ...     output_path=Path('output/ndvi_analysis.png')
    ... )
    """
    # Print statistics
    print_ndvi_statistics(ndvi_results, zones_data, zone_mapping)
    
    # Create visualization
    fig = visualize_ndvi_distribution(
        ndvi_results, zones_data, zone_mapping,
        output_path=output_path, dpi=dpi
    )
    
    return fig


def visualize_semantic_maps(
    semantic_results: Dict,
    ndvi_results: Dict,
    mgrg_results: Dict,
    embeddings_data: Dict,
    zones_data: Dict,
    zone_mapping: Dict,
    output_path: Optional[Path] = None,
    figsize: Optional[Tuple[int, int]] = None,
    dpi: int = 150,
    show_plot: bool = True
) -> Tuple[plt.Figure, Dict]:
    """
    Visualize semantic classification maps for all zones.
    
    Creates a comprehensive visualization showing:
    - NDVI original
    - MGRG segmentation
    - Semantic classification map
    
    Also generates colored maps for each zone.
    
    Parameters
    ----------
    semantic_results : dict
        Semantic classification results from classify_all_zones()
        Format: {zone_id: {'semantic_map': array, 'classifications': dict, ...}}
    ndvi_results : dict
        NDVI arrays for each zone
    mgrg_results : dict
        MGRG segmentation results
    embeddings_data : dict
        Embeddings data for each zone
    zones_data : dict
        Zone configuration data
    zone_mapping : dict
        Mapping from zone keys to zone IDs
    output_path : Path, optional
        Path to save the figure
    figsize : tuple, optional
        Figure size in inches (width, height). If None, auto-calculated.
    dpi : int, default=150
        Resolution for saved figure
    show_plot : bool, default=True
        Whether to display the plot
        
    Returns
    -------
    tuple
        (fig, colored_maps) where:
        - fig: matplotlib Figure object
        - colored_maps: dict with RGB colored maps per zone
        
    Examples
    --------
    >>> fig, colored_maps = visualize_semantic_maps(
    ...     semantic_results=semantic_results,
    ...     ndvi_results=ndvi_results,
    ...     mgrg_results=mgrg_results,
    ...     embeddings_data=embeddings_data,
    ...     zones_data=zones_data,
    ...     zone_mapping=zone_mapping,
    ...     output_path=Path('output/semantic_maps.png')
    ... )
    """
    from matplotlib.patches import Rectangle
    from src.classification.zero_shot_classifier import (
        SemanticClassifier, LAND_COVER_CLASSES, CLASS_COLORS
    )
    
    print("VISUALIZACIÓN DE MAPAS SEMÁNTICOS\n")
    
    # Generate colored semantic maps for all zones
    colored_maps = {}
    for zone_name, zone_id in zone_mapping.items():
        if zone_id not in semantic_results:
            continue
        
        zone_display = zones_data[zone_id]['config']['name']
        semantic_map = semantic_results[zone_id]['semantic_map']
        
        # Generate colored map
        classifier = SemanticClassifier(
            embeddings=embeddings_data[zone_name]['embeddings'],
            ndvi=ndvi_results[zone_id]
        )
        colored_map = classifier.generate_colored_map(semantic_map)
        colored_maps[zone_id] = colored_map
        
        print(f"{zone_display}:")
        print(f"  Shape: {semantic_map.shape}")
        print(f"  Clases únicas: {np.unique(semantic_map)}")
    
    # Count valid zones
    valid_zones = [(zn, zid) for zn, zid in zone_mapping.items() if zid in semantic_results]
    n_zones = len(valid_zones)
    
    if n_zones == 0:
        raise ValueError("No valid zones with semantic results")
    
    # Auto-calculate figsize if not provided
    if figsize is None:
        figsize = (20, 6 * n_zones)
    
    # Create comparative visualization: NDVI → Segmentation → Classification
    fig, axes = plt.subplots(n_zones, 3, figsize=figsize)
    
    # Handle single zone case
    if n_zones == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (zone_name, zone_id) in enumerate(valid_zones):
        zone_display = zones_data[zone_id]['config']['name']
        ndvi = ndvi_results[zone_id]
        segmentation = mgrg_results[zone_id]['labeled']
        colored_map = colored_maps[zone_id]
        num_regions = mgrg_results[zone_id]['num_regions']
        
        # Column 1: NDVI
        im1 = axes[idx, 0].imshow(ndvi, cmap='RdYlGn', vmin=-0.2, vmax=0.8)
        axes[idx, 0].set_title(
            f'{zone_display}\nNDVI Original',
            fontsize=12,
            fontweight='bold'
        )
        axes[idx, 0].axis('off')
        plt.colorbar(im1, ax=axes[idx, 0], label='NDVI', fraction=0.046, pad=0.04)
        
        # Column 2: Segmentation
        axes[idx, 1].imshow(segmentation, cmap='tab20', interpolation='nearest')
        axes[idx, 1].set_title(
            f'Segmentación MGRG\n({num_regions} regiones)',
            fontsize=12,
            fontweight='bold'
        )
        axes[idx, 1].axis('off')
        
        # Column 3: Semantic Map
        axes[idx, 2].imshow(colored_map)
        axes[idx, 2].set_title(
            'Mapa Semántico\n(Clasificado)',
            fontsize=12,
            fontweight='bold'
        )
        axes[idx, 2].axis('off')
    
    # Add legend to last row
    legend_elements = []
    for class_id, class_name in LAND_COVER_CLASSES.items():
        color = np.array(CLASS_COLORS[class_id]) / 255.0
        legend_elements.append(
            Rectangle((0, 0), 1, 1, fc=color, label=class_name)
        )
    
    axes[-1, 2].legend(
        handles=legend_elements,
        loc='center left',
        bbox_to_anchor=(1.05, 0.5),
        fontsize=9,
        frameon=True
    )
    
    plt.suptitle(
        'Pipeline Completo: NDVI → Segmentación → Clasificación Semántica',
        fontsize=16,
        fontweight='bold'
    )
    plt.tight_layout()
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"\nVisualización guardada en: {output_path}")
    
    if show_plot:
        plt.show()
    
    return fig, colored_maps


def visualize_class_statistics(
    semantic_results: Dict,
    zones_data: Dict,
    zone_mapping: Dict,
    output_dir: Optional[Path] = None,
    dpi: int = 300,
    show_plot: bool = True
) -> Dict[int, plt.Figure]:
    """
    Visualize statistics per class for each zone.
    
    Creates a 1x2 plot for each zone showing:
    - Bar plot: Area per class
    - Box plot: NDVI distribution per class
    
    Parameters
    ----------
    semantic_results : dict
        Semantic classification results from classify_all_zones()
    zones_data : dict
        Zone configuration data
    zone_mapping : dict
        Mapping from zone keys to zone IDs
    output_dir : Path, optional
        Directory to save figures. If None, figures are not saved.
    dpi : int, default=300
        Resolution for saved figures
    show_plot : bool, default=True
        Whether to display the plots
        
    Returns
    -------
    dict
        Dictionary mapping zone_id to matplotlib Figure objects
        
    Examples
    --------
    >>> figures = visualize_class_statistics(
    ...     semantic_results=semantic_results,
    ...     zones_data=zones_data,
    ...     zone_mapping=zone_mapping,
    ...     output_dir=OUTPUT_DIR,
    ...     dpi=300
    ... )
    """
    from src.classification.zero_shot_classifier import LAND_COVER_CLASSES, CLASS_COLORS
    
    print("VISUALIZACIÓN DE ESTADÍSTICAS POR CLASE (POR ZONA)\n")
    
    # Prepare class names and colors
    class_names = list(LAND_COVER_CLASSES.values())
    colors_normalized = [np.array(CLASS_COLORS[i]) / 255.0 for i in range(6)]
    
    figures = {}
    
    for zone_name, zone_id in zone_mapping.items():
        if zone_id not in semantic_results:
            continue
        
        zone_display = zones_data[zone_id]['config']['name']
        class_stats = semantic_results[zone_id]['class_stats']
        
        # Prepare data for plots
        areas = [
            class_stats[c]['area_ha'] if c in class_stats else 0 
            for c in LAND_COVER_CLASSES.values()
        ]
        
        # NDVI by class
        ndvi_by_class = {}
        for class_name in LAND_COVER_CLASSES.values():
            if class_name in class_stats and class_stats[class_name]['count'] > 0:
                ndvi_by_class[class_name] = [class_stats[class_name]['mean_ndvi']]
            else:
                ndvi_by_class[class_name] = [0]
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        
        # Plot 1: Bar plot - Area per class
        axes[0].bar(
            class_names, 
            areas, 
            color=colors_normalized, 
            edgecolor='black'
        )
        axes[0].set_ylabel('Área (hectáreas)', fontsize=12, fontweight='bold')
        axes[0].set_title(
            f'Área por Clase - {zone_display}', 
            fontsize=14, 
            fontweight='bold'
        )
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (name, area) in enumerate(zip(class_names, areas)):
            if area > 0:
                axes[0].text(
                    i, area + max(areas) * 0.02, 
                    f'{area:.1f}', 
                    ha='center', 
                    va='bottom',
                    fontsize=9
                )
        
        # Plot 2: Box plot - NDVI distribution per class
        box_data = [ndvi_by_class[c] for c in class_names]
        bp = axes[1].boxplot(
            box_data, 
            labels=class_names, 
            patch_artist=True
        )
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], colors_normalized):
            patch.set_facecolor(color)
            patch.set_edgecolor('black')
        
        axes[1].set_ylabel('NDVI', fontsize=12, fontweight='bold')
        axes[1].set_title(
            'Distribución de NDVI por Clase', 
            fontsize=14, 
            fontweight='bold'
        )
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(axis='y', alpha=0.3)
        axes[1].set_ylim(-0.2, 1.0)
        
        plt.tight_layout()
        
        # Save figure
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f'class_statistics_plots_{zone_name}.png'
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
            print(f"Estadísticas guardadas: {output_path}")
        
        # Show plot
        if show_plot:
            plt.show()
        
        figures[zone_id] = fig
    
    print(f"\nVisualización completada para {len(figures)} zonas")
    
    return figures


def visualize_class_distribution_summary(
    class_summary: pd.DataFrame,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (16, 6),
    dpi: int = 150,
    show_plot: bool = True
) -> plt.Figure:
    """
    Visualize overall class distribution summary across all zones.
    
    Creates a 1x2 plot showing:
    - Bar plot: Total area per class
    - Bar plot: Number of regions per class
    
    Parameters
    ----------
    class_summary : pd.DataFrame
        Summary DataFrame with columns: 'Clase', 'Área (ha)', 'Regiones'
        Typically created by grouping df_all_classes by 'Clase'
    output_path : Path, optional
        Path to save the figure
    figsize : tuple, default=(16, 6)
        Figure size in inches (width, height)
    dpi : int, default=150
        Resolution for saved figure
    show_plot : bool, default=True
        Whether to display the plot
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure object
        
    Examples
    --------
    >>> # Create summary from df_all_classes
    >>> class_summary = df_all_classes.groupby('Clase').agg({
    ...     'Área (ha)': 'sum',
    ...     'Regiones': 'sum'
    ... }).reset_index()
    >>> 
    >>> fig = visualize_class_distribution_summary(
    ...     class_summary=class_summary,
    ...     output_path=OUTPUT_DIR / 'class_distribution_summary.png'
    ... )
    """
    from src.classification.zero_shot_classifier import CLASS_COLORS
    
    print("\nGenerando gráfico de distribución de clases...\n")
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Prepare data
    class_areas = class_summary.set_index('Clase')['Área (ha)']
    class_regions = class_summary.set_index('Clase')['Regiones']
    
    # Normalize colors
    colors_normalized = [
        tuple(c / 255 for c in CLASS_COLORS[i]) 
        for i in range(6)
    ]
    
    # Plot 1: Area distribution per class
    axes[0].bar(
        range(len(class_areas)),
        class_areas.values,
        color=[colors_normalized[i] for i in range(len(class_areas))],
        edgecolor='black',
        alpha=0.8
    )
    axes[0].set_xticks(range(len(class_areas)))
    axes[0].set_xticklabels(class_areas.index, rotation=45, ha='right')
    axes[0].set_ylabel('Área Total (ha)', fontweight='bold', fontsize=12)
    axes[0].set_title(
        'Distribución de Área por Clase',
        fontweight='bold',
        fontsize=14
    )
    axes[0].grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(class_areas.values):
        axes[0].text(
            i, v, f'{v:.0f}',
            ha='center',
            va='bottom',
            fontweight='bold',
            fontsize=10
        )
    
    # Plot 2: Number of regions per class
    axes[1].bar(
        range(len(class_regions)),
        class_regions.values,
        color=[colors_normalized[i] for i in range(len(class_regions))],
        edgecolor='black',
        alpha=0.8
    )
    axes[1].set_xticks(range(len(class_regions)))
    axes[1].set_xticklabels(class_regions.index, rotation=45, ha='right')
    axes[1].set_ylabel('Número de Regiones', fontweight='bold', fontsize=12)
    axes[1].set_title(
        'Número de Regiones por Clase',
        fontweight='bold',
        fontsize=14
    )
    axes[1].grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(class_regions.values):
        axes[1].text(
            i, v, f'{v}',
            ha='center',
            va='bottom',
            fontweight='bold',
            fontsize=10
        )
    
    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Gráfico de distribución guardado en: {output_path}")

    if show_plot:
        plt.show()

    return fig


def visualize_class_statistics_all_zones(
    semantic_results: Dict,
    zones_data: Dict,
    zone_mapping: Dict,
    class_names: list,
    colors_normalized: list,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (18, 6),
    dpi: int = 300,
    show_plot: bool = True
) -> plt.Figure:
    """
    Visualize class statistics aggregated across all zones.

    Creates a 1x2 plot showing:
    - Bar plot: Area per class grouped by zone
    - Box plot: NDVI distribution per class across all zones

    Parameters
    ----------
    semantic_results : dict
        Semantic classification results from classify_all_zones()
    zones_data : dict
        Zone configuration data
    zone_mapping : dict
        Mapping from zone keys to zone IDs
    class_names : list
        List of land cover class names
    colors_normalized : list
        List of normalized RGB colors for each class
    output_path : Path, optional
        Path to save the figure
    figsize : tuple, default=(18, 6)
        Figure size in inches (width, height)
    dpi : int, default=300
        Resolution for saved figure
    show_plot : bool, default=True
        Whether to display the plot

    Returns
    -------
    matplotlib.figure.Figure
        The created figure object

    Examples
    --------
    >>> from src.classification.zero_shot_classifier import LAND_COVER_CLASSES, CLASS_COLORS
    >>> class_names = list(LAND_COVER_CLASSES.values())
    >>> colors_normalized = [np.array(CLASS_COLORS[i]) / 255.0 for i in range(6)]
    >>>
    >>> fig = visualize_class_statistics_all_zones(
    ...     semantic_results=semantic_results,
    ...     zones_data=zones_data,
    ...     zone_mapping=zone_mapping,
    ...     class_names=class_names,
    ...     colors_normalized=colors_normalized,
    ...     output_path=OUTPUT_DIR / 'class_statistics_all_zones.png'
    ... )
    """
    print("GENERANDO GRÁFICOS DE ESTADÍSTICAS\n")

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: Bar plot - Area per class grouped by zone
    areas_by_zone = {}

    for zone_name, zone_id in zone_mapping.items():
        if zone_id not in semantic_results:
            continue

        zone_display = zones_data[zone_id]['config']['name']
        class_stats = semantic_results[zone_id]['class_stats']

        areas = [
            class_stats[c]['area_ha'] if c in class_stats else 0
            for c in class_names
        ]
        areas_by_zone[zone_display] = areas

    # Plot grouped bars by zone
    x = np.arange(len(class_names))
    width = 0.25
    multiplier = 0

    # Define unique colors for each zone
    zone_colors = ['#90EE90', '#4682B4', '#FFD700']  # Light green, Steel blue, Gold

    for idx, (zone_name, areas) in enumerate(areas_by_zone.items()):
        offset = width * multiplier
        axes[0].bar(
            x + offset, areas, width,
            label=zone_name,
            color=zone_colors[idx % len(zone_colors)],
            alpha=0.8,
            edgecolor='black'
        )
        multiplier += 1

    axes[0].set_ylabel('Área (hectáreas)', fontsize=12, fontweight='bold')
    axes[0].set_title('Área por Clase - Todas las Zonas', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels(class_names, rotation=45, ha='right')
    axes[0].legend(loc='upper right')
    axes[0].grid(axis='y', alpha=0.3)

    # Plot 2: Box plot - NDVI distribution per class across all zones
    ndvi_by_class = {c: [] for c in class_names}

    for zone_name, zone_id in zone_mapping.items():
        if zone_id not in semantic_results:
            continue

        class_stats = semantic_results[zone_id]['class_stats']

        for class_name in class_names:
            if class_name in class_stats and class_stats[class_name]['count'] > 0:
                ndvi_by_class[class_name].append(class_stats[class_name]['mean_ndvi'])

    # Prepare boxplot data
    box_data = [
        ndvi_by_class[c] if len(ndvi_by_class[c]) > 0 else [0]
        for c in class_names
    ]

    bp = axes[1].boxplot(box_data, labels=class_names, patch_artist=True)

    # Color boxes
    for patch, color in zip(bp['boxes'], colors_normalized):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')
        patch.set_alpha(0.8)

    axes[1].set_ylabel('NDVI', fontsize=12, fontweight='bold')
    axes[1].set_title('Distribución de NDVI por Clase', fontsize=14, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Gráficos guardados en: {output_path}")

    if show_plot:
        plt.show()

    return fig


def visualize_dynamic_world_comparison(
    dw_results: Dict,
    semantic_results: Dict,
    zones_data: Dict,
    zone_mapping: Dict,
    our_to_dw_mapping: Dict,
    dw_colors: Dict,
    output_dir: Path,
    dpi: int = 300,
    show_plot: bool = True
) -> Dict[int, plt.Figure]:
    """
    Visualize comparison between semantic classification and Dynamic World.

    Creates a 1x3 plot for each zone showing:
    - Our semantic map (colored)
    - Dynamic World reference map (colored)
    - Agreement/disagreement map (green=agree, red=disagree)

    Parameters
    ----------
    dw_results : dict
        Results from cross_validate_with_dynamic_world()
        Format: {zone_id: {'dw_mask': array, 'agreements': dict, 'colored_map': array}}
    semantic_results : dict
        Semantic classification results
    zones_data : dict
        Zone configuration data
    zone_mapping : dict
        Mapping from zone keys to zone IDs
    our_to_dw_mapping : dict
        Mapping from our classes to Dynamic World classes
        Example: {0: 0, 1: 6, 2: 7, 3: 4, 4: 4, 5: 2}
    dw_colors : dict
        Dynamic World class colors (RGB 0-255)
        Example: {0: (65, 155, 223), 1: (57, 125, 73), ...}
    output_dir : Path
        Directory to save comparison figures
    dpi : int, default=300
        Resolution for saved figures
    show_plot : bool, default=True
        Whether to display the plots

    Returns
    -------
    dict
        Dictionary mapping zone_id to matplotlib Figure objects

    Examples
    --------
    >>> # Define mappings
    >>> our_to_dw = {0: 0, 1: 6, 2: 7, 3: 4, 4: 4, 5: 2}
    >>> dw_colors = {
    ...     0: (65, 155, 223),   # Water
    ...     1: (57, 125, 73),    # Trees
    ...     2: (136, 176, 83),   # Grass
    ...     4: (228, 150, 53),   # Crops
    ...     6: (196, 40, 27),    # Built
    ...     7: (165, 155, 143)   # Bare
    ... }
    >>>
    >>> figures = visualize_dynamic_world_comparison(
    ...     dw_results, semantic_results, zones_data, zone_mapping,
    ...     our_to_dw, dw_colors, OUTPUT_DIR
    ... )
    """
    print("\nGENERANDO VISUALIZACIONES DE COMPARACIÓN\n")

    figures = {}

    for zone_name, zone_id in zone_mapping.items():
        if zone_id not in dw_results:
            continue

        zone_display = zones_data[zone_id]['config']['name']
        dw_mask = dw_results[zone_id]['dw_mask']
        colored_map = dw_results[zone_id]['colored_map']
        semantic_map = semantic_results[zone_id]['semantic_map']
        agreements = dw_results[zone_id]['agreements']

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        # Plot 1: Our semantic map
        axes[0].imshow(colored_map)
        axes[0].set_title(
            f'Nuestro Mapa Semántico\n{zone_display}',
            fontsize=14,
            fontweight='bold'
        )
        axes[0].axis('off')

        # Plot 2: Dynamic World colored map
        dw_colored = np.zeros((*dw_mask.shape, 3), dtype=np.uint8)
        for class_id, color in dw_colors.items():
            mask = (dw_mask == class_id)
            dw_colored[mask] = color

        axes[1].imshow(dw_colored)
        axes[1].set_title(
            'Dynamic World (Google)',
            fontsize=14,
            fontweight='bold'
        )
        axes[1].axis('off')

        # Plot 3: Agreement/disagreement map
        # Convert semantic map to Dynamic World format
        semantic_map_dw_format = np.zeros_like(semantic_map)
        for our_class, dw_class in our_to_dw_mapping.items():
            semantic_map_dw_format[semantic_map == our_class] = dw_class

        # Create difference map
        agreement_mask = (semantic_map_dw_format == dw_mask)
        diff_map = np.zeros((*semantic_map.shape, 3), dtype=np.uint8)
        diff_map[agreement_mask] = [0, 255, 0]      # Green = Agreement
        diff_map[~agreement_mask] = [255, 0, 0]     # Red = Disagreement

        axes[2].imshow(diff_map)
        axes[2].set_title(
            f'Acuerdo/Desacuerdo\n({agreements["overall"]:.1%})',
            fontsize=14,
            fontweight='bold'
        )
        axes[2].axis('off')

        plt.tight_layout()

        # Save figure
        output_path = output_dir / f'dynamic_world_comparison_{zone_name}.png'
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')

        if show_plot:
            plt.show()

        print(f"  ✓ Comparación guardada: {output_path.name}")

        figures[zone_id] = fig

    print(f"\n✓ Visualizaciones completadas para {len(figures)} zonas")

    return figures
