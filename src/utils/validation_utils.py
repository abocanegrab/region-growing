"""
Validation utilities for segmentation results.

This module provides functions for validating segmentation results
against ground truth data (Dynamic World or synthetic).
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple, Optional
import rasterio
from scipy.ndimage import zoom


# Validation classes (5 classes for simplification)
VALIDATION_CLASSES = {
    0: 'Water',
    1: 'Crop',
    2: 'Urban',
    3: 'Bare Soil',
    4: 'Other'
}


def map_to_validation_classes(segmentation: np.ndarray, ndvi: np.ndarray) -> np.ndarray:
    """
    Map segmentation to 5 validation classes based on NDVI.
    
    Parameters
    ----------
    segmentation : np.ndarray
        Labeled segmentation array
    ndvi : np.ndarray
        NDVI array
        
    Returns
    -------
    np.ndarray
        Mapped classification (0-4)
        
    Examples
    --------
    >>> mapped = map_to_validation_classes(segmentation, ndvi)
    >>> print(np.unique(mapped))
    [0 1 2 3 4]
    """
    mapped = np.zeros_like(segmentation, dtype=np.uint8)
    
    for region_id in np.unique(segmentation):
        if region_id == 0:
            continue
        
        mask = (segmentation == region_id)
        region_ndvi = ndvi[mask]
        mean_ndvi = np.mean(region_ndvi)
        
        # Classify based on NDVI
        if mean_ndvi < 0.1:
            mapped[mask] = 0  # Water
        elif mean_ndvi < 0.3:
            mapped[mask] = 3  # Bare Soil
        elif mean_ndvi < 0.6:
            mapped[mask] = 1  # Crop
        else:
            mapped[mask] = 1  # Crop (vigorous)
    
    return mapped


def load_ground_truth(
    zone_name: str,
    dynamic_world_path: Path,
    target_shape: Tuple[int, int]
) -> Tuple[np.ndarray, bool]:
    """
    Load ground truth data from Dynamic World or generate synthetic.
    
    Parameters
    ----------
    zone_name : str
        Name of the zone
    dynamic_world_path : Path
        Path to Dynamic World data directory
    target_shape : tuple
        Target shape (height, width) for resizing
        
    Returns
    -------
    tuple
        (ground_truth_mask, is_synthetic)
        
    Examples
    --------
    >>> gt_mask, is_synthetic = load_ground_truth('mexicali', dw_path, (512, 512))
    >>> print(f"Synthetic: {is_synthetic}")
    """
    dw_file = dynamic_world_path / f'{zone_name}_dw.tif'
    
    if dw_file.exists():
        print(f"  Ground Truth: Dynamic World (real)")
        with rasterio.open(dw_file) as src:
            gt_mask = src.read(1)
        
        # Resize if necessary
        if gt_mask.shape != target_shape:
            zoom_factors = (
                target_shape[0] / gt_mask.shape[0],
                target_shape[1] / gt_mask.shape[1]
            )
            gt_mask = zoom(gt_mask, zoom_factors, order=0)
            print(f"  Ground Truth redimensionado: {gt_mask.shape}")
        
        is_synthetic = False
    else:
        print(f"  Ground Truth: Sintético (basado en NDVI)")
        # Import here to avoid circular dependency
        from src.utils.dynamic_world_downloader import load_or_generate_ground_truth
        gt_mask, is_synthetic = load_or_generate_ground_truth(
            zone_name, 
            None,  # Will be handled by the function
            dynamic_world_path, 
            use_synthetic=True
        )
    
    print(f"  Ground Truth shape: {gt_mask.shape}")
    return gt_mask, is_synthetic


def validate_all_zones(
    zones_data: Dict,
    zone_mapping: Dict,
    segmentation_results: Dict,
    mgrg_results: Dict,
    ndvi_results: Dict,
    dynamic_world_path: Path,
    results_path: Path,
    num_classes: int = 5,
    save_confusion_matrices: bool = True
) -> Dict:
    """
    Validate all zones against ground truth.
    
    Parameters
    ----------
    zones_data : dict
        Zone configuration data
    zone_mapping : dict
        Mapping from zone keys to zone IDs
    segmentation_results : dict
        Classic RG results
    mgrg_results : dict
        MGRG results
    ndvi_results : dict
        NDVI arrays
    dynamic_world_path : Path
        Path to Dynamic World data
    results_path : Path
        Path to save results
    num_classes : int, default=5
        Number of validation classes
    save_confusion_matrices : bool, default=True
        Whether to save confusion matrix plots
        
    Returns
    -------
    dict
        Validation results for all zones
        
    Examples
    --------
    >>> validation_results = validate_all_zones(
    ...     zones_data, zone_mapping, segmentation_results,
    ...     mgrg_results, ndvi_results, dw_path, results_path
    ... )
    """
    # Import validation functions
    from src.utils.validation_helpers import validate_zone, display_zone_metrics
    
    validation_results = {}
    
    for zone_name, zone_id in zone_mapping.items():
        if zone_id not in segmentation_results or zone_id not in mgrg_results:
            print(f"\n⚠️ Saltando {zone_name}: no hay resultados de segmentación")
            continue
        
        print(f"\n{'='*80}")
        print(f"VALIDANDO: {zone_name.upper()}")
        print(f"{'='*80}")
        
        # Get segmentations
        classic_seg = segmentation_results[zone_id]['labeled']
        mgrg_seg = mgrg_results[zone_id]['labeled']
        ndvi = ndvi_results[zone_id]
        
        print(f"  Classic RG: {classic_seg.shape}, {segmentation_results[zone_id]['num_regions']} regiones")
        print(f"  MGRG: {mgrg_seg.shape}, {mgrg_results[zone_id]['num_regions']} regiones")
        
        # Load ground truth
        gt_mask, is_synthetic = load_ground_truth(
            zone_name,
            dynamic_world_path,
            classic_seg.shape
        )
        
        # Map segmentations to validation classes
        print("\n  Mapeando segmentaciones a clases de validación...")
        classic_mapped = map_to_validation_classes(classic_seg, ndvi)
        mgrg_mapped = map_to_validation_classes(mgrg_seg, ndvi)
        
        # Validate
        results = validate_zone(
            zone_name,
            classic_mapped,
            mgrg_mapped,
            gt_mask,
            num_classes=num_classes,
            verbose=False
        )
        
        # Add original region counts
        results['classic_rg']['num_regions'] = segmentation_results[zone_id]['num_regions']
        results['mgrg']['num_regions'] = mgrg_results[zone_id]['num_regions']
        results['is_synthetic'] = is_synthetic
        
        validation_results[zone_name] = results
        
        # Display metrics
        display_zone_metrics(results, zone_name)
        
        # Save confusion matrices
        if save_confusion_matrices:
            visualize_confusion_matrices(
                results,
                zone_name,
                list(VALIDATION_CLASSES.values()),
                results_path
            )
    
    print("\n" + "="*80)
    print("✓ VALIDACIÓN COMPLETADA PARA TODAS LAS ZONAS")
    print("="*80 + "\n")
    
    return validation_results


def visualize_confusion_matrices(
    results: Dict,
    zone_name: str,
    class_names: list,
    output_path: Path,
    dpi: int = 300
) -> None:
    """
    Visualize confusion matrices for both methods.
    
    Parameters
    ----------
    results : dict
        Validation results from validate_zone()
    zone_name : str
        Name of the zone
    class_names : list
        List of class names
    output_path : Path
        Directory to save the figure
    dpi : int, default=300
        Resolution for saved figure
        
    Examples
    --------
    >>> visualize_confusion_matrices(
    ...     results, 'mexicali', class_names, output_path
    ... )
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, (method_name, method_key) in enumerate([('Classic RG', 'classic_rg'), ('MGRG', 'mgrg')]):
        cm = results[method_key]['confusion_matrix']
        im = axes[idx].imshow(cm, cmap='Blues')
        
        axes[idx].set_xticks(np.arange(len(class_names)))
        axes[idx].set_yticks(np.arange(len(class_names)))
        axes[idx].set_xticklabels(class_names, rotation=45, ha='right')
        axes[idx].set_yticklabels(class_names)
        axes[idx].set_xlabel('Predicted', fontweight='bold')
        axes[idx].set_ylabel('Ground Truth', fontweight='bold')
        axes[idx].set_title(
            f'{method_name}\nmIoU: {results[method_key]["miou"]:.4f}',
            fontweight='bold'
        )
        
        # Add values in cells
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                text_color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
                axes[idx].text(
                    j, i, f'{cm[i, j]}',
                    ha='center', va='center',
                    color=text_color, fontsize=9
                )
        
        plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
    
    plt.suptitle(
        f'Confusion Matrices - {zone_name.capitalize()}',
        fontsize=14,
        fontweight='bold'
    )
    plt.tight_layout()
    
    output_file = output_path / f'confusion_matrices_{zone_name}.png'
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.show()
    
    print(f"  ✓ Confusion matrices guardadas: {output_file}")


def print_validation_classes() -> None:
    """
    Print validation classes in a formatted way.
    
    Examples
    --------
    >>> print_validation_classes()
    Clases de validación:
      0: Water
      1: Crop
      ...
    """
    print("\nClases de validación:")
    for class_id, class_name in VALIDATION_CLASSES.items():
        print(f"  {class_id}: {class_name}")



def calculate_aggregate_metrics(
    validation_results: Dict,
    use_display: bool = True
) -> Tuple[Dict, Dict, Dict]:
    """
    Calculate aggregate metrics with mean and standard deviation.
    
    Parameters
    ----------
    validation_results : dict
        Validation results from validate_all_zones()
    use_display : bool, default=True
        If True, uses IPython display for DataFrame output
        
    Returns
    -------
    tuple
        (aggregated_df, improvements_df, zone_stats_df)
        
    Examples
    --------
    >>> agg_df, imp_df, zone_df = calculate_aggregate_metrics(validation_results)
    """
    import pandas as pd
    
    # Metrics to aggregate
    metrics_to_aggregate = {
        'mIoU': 'miou',
        'Weighted mIoU': 'weighted_miou',
        'Macro F1': 'macro_f1',
        'Precision': 'macro_precision',
        'Recall': 'macro_recall',
        'Pixel Acc': 'pixel_accuracy'
    }
    
    # 1. AGGREGATE METRICS WITH STD
    print("\n" + "="*100)
    print("MÉTRICAS AGREGADAS (Promedio ± Desviación Estándar)")
    print("="*100 + "\n")
    
    aggregated_data = []
    for method_name, method_key in [('Classic RG', 'classic_rg'), ('MGRG', 'mgrg')]:
        row = {'Método': method_name}
        
        for display_name, metric_key in metrics_to_aggregate.items():
            values = [validation_results[zone][method_key][metric_key] 
                     for zone in validation_results.keys()]
            mean_val = np.mean(values)
            std_val = np.std(values)
            row[display_name] = f"{mean_val:.4f} ± {std_val:.4f}"
        
        # Add average number of regions
        num_regions = [validation_results[zone][method_key]['num_regions'] 
                      for zone in validation_results.keys()]
        row['Regiones'] = f"{np.mean(num_regions):.0f} ± {np.std(num_regions):.0f}"
        
        aggregated_data.append(row)
    
    df_aggregated = pd.DataFrame(aggregated_data)
    
    if use_display:
        try:
            from IPython.display import display
            display(df_aggregated)
        except ImportError:
            print(df_aggregated.to_string(index=False))
    else:
        print(df_aggregated.to_string(index=False))
    
    # 2. IMPROVEMENT ANALYSIS
    print("\n" + "="*100)
    print("ANÁLISIS DE MEJORA (MGRG vs Classic RG)")
    print("="*100 + "\n")
    
    improvements = {}
    for metric_display, metric_key in metrics_to_aggregate.items():
        zone_improvements = []
        for zone in validation_results.keys():
            classic_val = validation_results[zone]['classic_rg'][metric_key]
            mgrg_val = validation_results[zone]['mgrg'][metric_key]
            
            if classic_val > 0:
                improvement = ((mgrg_val - classic_val) / classic_val) * 100
                zone_improvements.append(improvement)
            else:
                # If classic is 0, calculate absolute improvement
                zone_improvements.append(mgrg_val * 100)
        
        improvements[metric_display] = {
            'mean': np.mean(zone_improvements),
            'std': np.std(zone_improvements),
            'min': np.min(zone_improvements),
            'max': np.max(zone_improvements)
        }
    
    improvement_data = []
    for metric, stats in improvements.items():
        improvement_data.append({
            'Métrica': metric,
            'Mejora Media': f"{stats['mean']:+.1f}%",
            'Desv. Est.': f"±{stats['std']:.1f}%",
            'Rango': f"[{stats['min']:+.1f}%, {stats['max']:+.1f}%]"
        })
    
    df_improvements = pd.DataFrame(improvement_data)
    
    if use_display:
        try:
            from IPython.display import display
            display(df_improvements)
        except ImportError:
            print(df_improvements.to_string(index=False))
    else:
        print(df_improvements.to_string(index=False))
    
    # 3. ZONE STATISTICS
    print("\n" + "="*100)
    print("ESTADÍSTICAS DETALLADAS POR ZONA")
    print("="*100 + "\n")
    
    zone_stats = []
    for zone_name in validation_results.keys():
        classic = validation_results[zone_name]['classic_rg']
        mgrg = validation_results[zone_name]['mgrg']
        
        # Calculate mIoU improvement
        if classic['miou'] > 0:
            miou_improvement = ((mgrg['miou'] - classic['miou']) / classic['miou']) * 100
        else:
            miou_improvement = float('inf') if mgrg['miou'] > 0 else 0
        
        # Region reduction
        region_reduction = (1 - mgrg['num_regions'] / classic['num_regions']) * 100
        
        zone_stats.append({
            'Zona': zone_name.capitalize(),
            'Classic Regiones': classic['num_regions'],
            'MGRG Regiones': mgrg['num_regions'],
            'Reducción': f"{region_reduction:.1f}%",
            'Classic mIoU': f"{classic['miou']:.4f}",
            'MGRG mIoU': f"{mgrg['miou']:.4f}",
            'Mejora mIoU': f"{miou_improvement:+.1f}%" if miou_improvement != float('inf') else "N/A",
            'Ground Truth': 'Real' if not validation_results[zone_name].get('is_synthetic', True) else 'Sintético'
        })
    
    df_zone_stats = pd.DataFrame(zone_stats)
    
    if use_display:
        try:
            from IPython.display import display
            display(df_zone_stats)
        except ImportError:
            print(df_zone_stats.to_string(index=False))
    else:
        print(df_zone_stats.to_string(index=False))
    
    # 4. FINAL SUMMARY
    print("\n" + "="*100)
    print("RESUMEN FINAL DE VALIDACIÓN")
    print("="*100)
    
    classic_miou_values = [r['classic_rg']['miou'] for r in validation_results.values()]
    mgrg_miou_values = [r['mgrg']['miou'] for r in validation_results.values()]
    classic_regions = [r['classic_rg']['num_regions'] for r in validation_results.values()]
    mgrg_regions = [r['mgrg']['num_regions'] for r in validation_results.values()]
    
    classic_miou_mean = np.mean(classic_miou_values)
    classic_miou_std = np.std(classic_miou_values)
    mgrg_miou_mean = np.mean(mgrg_miou_values)
    mgrg_miou_std = np.std(mgrg_miou_values)
    
    print(f"\nResultados en {len(validation_results)} zonas agrícolas:\n")
    print(f"Classic RG:")
    print(f"  • mIoU: {classic_miou_mean:.4f} ± {classic_miou_std:.4f}")
    print(f"  • Regiones promedio: {np.mean(classic_regions):.0f}")
    print(f"\nMGRG:")
    print(f"  • mIoU: {mgrg_miou_mean:.4f} ± {mgrg_miou_std:.4f}")
    print(f"  • Regiones promedio: {np.mean(mgrg_regions):.0f}")
    
    if classic_miou_mean > 0:
        overall_improvement = ((mgrg_miou_mean - classic_miou_mean) / classic_miou_mean) * 100
        print(f"\nMejora general: {overall_improvement:+.1f}%")
        if overall_improvement > 0:
            print(f"✓ MGRG supera a Classic RG en mIoU promedio")
        else:
            print(f"✓ Classic RG supera a MGRG en mIoU promedio")
    
    print("="*100 + "\n")
    
    return df_aggregated, df_improvements, df_zone_stats



def analyze_relative_improvements(
    validation_results: Dict,
    results_path: Path,
    use_display: bool = True
) -> Tuple:
    """
    Analyze relative improvements of MGRG vs Classic RG with statistical tests.
    
    Performs comprehensive analysis including:
    - Relative improvement calculations
    - Paired t-tests for statistical significance
    - Visualization of improvements
    - Summary statistics
    
    Parameters
    ----------
    validation_results : dict
        Validation results from validate_all_zones()
    results_path : Path
        Path to save visualizations
    use_display : bool, default=True
        If True, uses IPython display for DataFrame output
        
    Returns
    -------
    tuple
        (comparison_df, statistical_df, improvements_dict)
        
    Examples
    --------
    >>> comp_df, stats_df, improvements = analyze_relative_improvements(
    ...     validation_results, results_path
    ... )
    """
    import pandas as pd
    from scipy import stats as scipy_stats
    
    # Metrics to compare
    metrics_to_compare = [
        ('mIoU', 'miou'),
        ('Weighted mIoU', 'weighted_miou'),
        ('Macro F1', 'macro_f1'),
        ('Precision', 'macro_precision'),
        ('Recall', 'macro_recall'),
        ('Pixel Accuracy', 'pixel_accuracy')
    ]
    
    # 1. RELATIVE IMPROVEMENT ANALYSIS
    print("\n" + "="*100)
    print("ANÁLISIS DE MEJORA RELATIVA: MGRG vs Classic RG")
    print("="*100 + "\n")
    
    comparison_data = []
    for metric_name, metric_key in metrics_to_compare:
        # Extract values from all zones
        classic_values = [validation_results[zone]['classic_rg'][metric_key] 
                         for zone in validation_results.keys()]
        mgrg_values = [validation_results[zone]['mgrg'][metric_key] 
                      for zone in validation_results.keys()]
        
        # Calculate statistics
        classic_mean = np.mean(classic_values)
        classic_std = np.std(classic_values)
        mgrg_mean = np.mean(mgrg_values)
        mgrg_std = np.std(mgrg_values)
        
        # Calculate relative improvement
        if classic_mean > 0:
            improvement = ((mgrg_mean - classic_mean) / classic_mean) * 100
            improvement_str = f"{improvement:+.1f}%"
        else:
            improvement = np.nan
            improvement_str = "N/A"
        
        # Determine winner
        if not np.isnan(improvement):
            if improvement > 5:
                winner = "✅ MGRG"
            elif improvement < -5:
                winner = "⚠️ Classic RG"
            else:
                winner = "≈ Similar"
        else:
            winner = "N/A"
        
        comparison_data.append({
            'Métrica': metric_name,
            'Classic RG': f"{classic_mean:.4f} ± {classic_std:.4f}",
            'MGRG': f"{mgrg_mean:.4f} ± {mgrg_std:.4f}",
            'Mejora (%)': improvement_str,
            'Ganador': winner
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    
    if use_display:
        try:
            from IPython.display import display
            display(df_comparison)
        except ImportError:
            print(df_comparison.to_string(index=False))
    else:
        print(df_comparison.to_string(index=False))
    
    # 2. STATISTICAL SIGNIFICANCE ANALYSIS
    print("\n" + "="*100)
    print("ANÁLISIS ESTADÍSTICO (Prueba t pareada)")
    print("="*100 + "\n")
    
    statistical_results = []
    for metric_name, metric_key in metrics_to_compare:
        classic_values = [validation_results[zone]['classic_rg'][metric_key] 
                         for zone in validation_results.keys()]
        mgrg_values = [validation_results[zone]['mgrg'][metric_key] 
                      for zone in validation_results.keys()]
        
        # Paired t-test
        if len(classic_values) >= 2:  # Need at least 2 samples
            t_stat, p_value = scipy_stats.ttest_rel(mgrg_values, classic_values)
            
            # Interpret significance
            if p_value < 0.01:
                significance = "*** (p<0.01)"
            elif p_value < 0.05:
                significance = "** (p<0.05)"
            elif p_value < 0.10:
                significance = "* (p<0.10)"
            else:
                significance = "ns (no significativo)"
            
            statistical_results.append({
                'Métrica': metric_name,
                't-statistic': f"{t_stat:.3f}",
                'p-value': f"{p_value:.4f}",
                'Significancia': significance
            })
    
    if statistical_results:
        df_stats = pd.DataFrame(statistical_results)
        
        if use_display:
            try:
                from IPython.display import display
                display(df_stats)
            except ImportError:
                print(df_stats.to_string(index=False))
        else:
            print(df_stats.to_string(index=False))
        
        print("\nInterpretación:")
        print("  *** p<0.01: Altamente significativo")
        print("  **  p<0.05: Significativo")
        print("  *   p<0.10: Marginalmente significativo")
        print("  ns: No significativo")
    else:
        df_stats = None
        print("⚠️ No hay suficientes zonas para análisis estadístico (mínimo 2)")
    
    # 3. VISUALIZATION OF IMPROVEMENTS
    print("\n" + "="*100)
    print("VISUALIZACIÓN DE MEJORAS POR MÉTRICA")
    print("="*100 + "\n")
    
    # Prepare data for visualization
    metrics_names = []
    improvements_values = []
    
    for metric_name, metric_key in metrics_to_compare:
        classic_values = [validation_results[zone]['classic_rg'][metric_key] 
                         for zone in validation_results.keys()]
        mgrg_values = [validation_results[zone]['mgrg'][metric_key] 
                      for zone in validation_results.keys()]
        
        classic_mean = np.mean(classic_values)
        mgrg_mean = np.mean(mgrg_values)
        
        if classic_mean > 0:
            improvement = ((mgrg_mean - classic_mean) / classic_mean) * 100
            metrics_names.append(metric_name)
            improvements_values.append(improvement)
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['green' if x > 0 else 'red' for x in improvements_values]
    bars = ax.bar(metrics_names, improvements_values, color=colors, alpha=0.7, edgecolor='black')
    
    # Add line at y=0
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    # Add values on bars
    for bar, value in zip(bars, improvements_values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2., height,
            f'{value:+.1f}%',
            ha='center', va='bottom' if height > 0 else 'top',
            fontweight='bold', fontsize=10
        )
    
    ax.set_ylabel('Mejora Relativa (%)', fontweight='bold', fontsize=12)
    ax.set_title('Mejora de MGRG sobre Classic RG por Métrica', fontweight='bold', fontsize=14)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save figure
    fig_path = results_path / 'relative_improvement_by_metric.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f'Gráfico guardado en: {fig_path}')
    
    # 4. SUMMARY OF IMPROVEMENTS
    print("\n" + "="*100)
    print("RESUMEN DE MEJORAS")
    print("="*100)
    
    # Count positive and negative improvements
    positive_improvements = sum(1 for x in improvements_values if x > 0)
    negative_improvements = sum(1 for x in improvements_values if x < 0)
    total_metrics = len(improvements_values)
    
    print(f"\nMétricas analizadas: {total_metrics}")
    print(f"  • MGRG superior: {positive_improvements} métricas ({positive_improvements/total_metrics*100:.0f}%)")
    print(f"  • Classic RG superior: {negative_improvements} métricas ({negative_improvements/total_metrics*100:.0f}%)")
    
    # Best and worst improvement
    if improvements_values:
        best_improvement = max(improvements_values)
        worst_improvement = min(improvements_values)
        best_metric = metrics_names[improvements_values.index(best_improvement)]
        worst_metric = metrics_names[improvements_values.index(worst_improvement)]
        
        print(f"\nMejor mejora:")
        print(f"  • {best_metric}: {best_improvement:+.1f}%")
        print(f"\nPeor resultado:")
        print(f"  • {worst_metric}: {worst_improvement:+.1f}%")
        
        # Average improvement
        avg_improvement = np.mean(improvements_values)
        print(f"\nMejora promedio general: {avg_improvement:+.1f}%")
    
    print("="*100 + "\n")
    
    # Create improvements dictionary
    improvements_dict = {
        'metrics_names': metrics_names,
        'improvements_values': improvements_values,
        'positive_count': positive_improvements,
        'negative_count': negative_improvements,
        'avg_improvement': avg_improvement if improvements_values else 0
    }
    
    return df_comparison, df_stats, improvements_dict


def cross_validate_with_dynamic_world(
    semantic_results: Dict,
    zones_data: Dict,
    zone_mapping: Dict,
    colored_maps: Dict,
    dynamic_world_path: Path,
    target_agreement: float = 0.70
) -> Tuple[Dict, list]:
    """
    Cross-validate semantic classification results with Dynamic World data.

    Loads Dynamic World reference data, resizes if necessary, and calculates
    class-wise and overall agreement metrics.

    Parameters
    ----------
    semantic_results : dict
        Semantic classification results from classify_all_zones()
    zones_data : dict
        Zone configuration data
    zone_mapping : dict
        Mapping from zone keys to zone IDs
    colored_maps : dict
        Dictionary with RGB colored maps per zone
    dynamic_world_path : Path
        Path to Dynamic World data directory
    target_agreement : float, default=0.70
        Target agreement threshold (70%)

    Returns
    -------
    tuple
        (dw_results, all_agreements) where:
        - dw_results: dict with Dynamic World masks, agreements, and colored maps
        - all_agreements: list of overall agreement values for all zones

    Examples
    --------
    >>> dw_results, agreements = cross_validate_with_dynamic_world(
    ...     semantic_results, zones_data, zone_mapping, colored_maps,
    ...     Path('data/dynamic_world'), target_agreement=0.70
    ... )
    >>> print(f"Average agreement: {np.mean(agreements):.1%}")
    """
    from src.classification.zero_shot_classifier import (
        LAND_COVER_CLASSES, cross_validate_with_dynamic_world as cv_dw
    )

    print("CROSS-VALIDATION CON DYNAMIC WORLD\n")

    dw_results = {}
    all_agreements = []

    for zone_name, zone_id in zone_mapping.items():
        if zone_id not in semantic_results:
            continue

        zone_display = zones_data[zone_id]['config']['name']
        print(f"Procesando: {zone_display}")

        # Load Dynamic World data
        dw_path = dynamic_world_path / f"{zone_name}_dw.tif"

        if not dw_path.exists():
            print(f"  ⚠️ Archivo Dynamic World no encontrado: {dw_path}")
            continue

        try:
            with rasterio.open(dw_path) as src:
                dw_mask = src.read(1)

            print(f"  Dynamic World shape: {dw_mask.shape}")
            print(f"  Unique classes: {np.unique(dw_mask)}")

            # Get semantic map for this zone
            semantic_map = semantic_results[zone_id]['semantic_map']

            # Resize Dynamic World to match semantic map if needed
            if dw_mask.shape != semantic_map.shape:
                zoom_factors = (
                    semantic_map.shape[0] / dw_mask.shape[0],
                    semantic_map.shape[1] / dw_mask.shape[1]
                )
                dw_mask = zoom(dw_mask, zoom_factors, order=0)
                print(f"  Resized Dynamic World to: {dw_mask.shape}")

            # Calculate cross-validation agreements
            agreements = cv_dw(semantic_map, dw_mask)

            print("\n  === Resultados de Cross-Validation ===")
            for class_name in LAND_COVER_CLASSES.values():
                if class_name in agreements:
                    agreement = agreements[class_name]
                    print(f"  {class_name:35s}: {agreement:.1%} agreement")

            print(f"\n  Overall Agreement: {agreements['overall']:.1%}")

            # Check if target is met
            status = "✓ OK" if agreements['overall'] >= target_agreement else "⚠️ Below target"
            print(f"  Target (>{target_agreement:.0%}): {status}\n")

            # Store results
            dw_results[zone_id] = {
                'dw_mask': dw_mask,
                'agreements': agreements,
                'colored_map': colored_maps[zone_id]
            }
            all_agreements.append(agreements['overall'])

        except Exception as e:
            print(f"  ✗ Error procesando {zone_display}: {str(e)}\n")
            continue

    # Calculate average agreement across all zones
    if all_agreements:
        avg_agreement = np.mean(all_agreements)
        print(f"\n{'='*80}")
        print(f"PROMEDIO DE ACUERDO EN TODAS LAS ZONAS: {avg_agreement:.1%}")
        print(f"{'='*80}\n")

    return dw_results, all_agreements
