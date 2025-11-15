"""
Segmentation utilities for region growing algorithms.

This module provides helper functions for running segmentation algorithms
across multiple zones and managing results.
"""
import time
import numpy as np
from typing import Dict, List, Tuple, Optional


def run_classic_rg_segmentation(
    zones_data: Dict,
    ndvi_results: Dict,
    seeds_by_zone: Dict,
    threshold: float = 0.1,
    min_region_size: int = 50
) -> Dict:
    """
    Run Classic Region Growing segmentation on multiple zones.

    This function executes traditional NDVI-based region growing across
    multiple agricultural zones.

    Parameters
    ----------
    zones_data : dict
        Dictionary with zone configuration data
        Format: {zone_id: {'config': {'name': str, ...}, ...}}
    ndvi_results : dict
        Dictionary with NDVI arrays for each zone
        Format: {zone_id: ndvi_array}
    seeds_by_zone : dict
        Seeds for each zone
        Format: {zone_id: [(y1, x1), (y2, x2), ...]}
    threshold : float, default=0.1
        NDVI difference threshold for region growing
    min_region_size : int, default=50
        Minimum number of pixels for a valid region

    Returns
    -------
    dict
        Dictionary with Classic RG results for each zone:
        {
            zone_id: {
                'labeled': np.ndarray,
                'num_regions': int,
                'regions_info': list,
                'elapsed_time': float,
                'algorithm': ClassicRegionGrowing
            }
        }

    Examples
    --------
    >>> from src.algorithms.classic_region_growing import ClassicRegionGrowing
    >>> segmentation_results = run_classic_rg_segmentation(
    ...     zones_data=zones_data,
    ...     ndvi_results=ndvi_results,
    ...     seeds_by_zone=seeds_by_zone,
    ...     threshold=0.1,
    ...     min_region_size=50
    ... )
    >>> print(f"Processed {len(segmentation_results)} zones")
    """
    from src.algorithms.classic_region_growing import ClassicRegionGrowing

    print("\n" + "="*80)
    print("SEGMENTACIÓN CON CLASSIC REGION GROWING")
    print("="*80)
    print("\nParámetros:")
    print(f"  • Threshold (NDVI): {threshold}")
    print(f"  • Min region size: {min_region_size} píxeles")
    print("-"*80 + "\n")

    segmentation_results = {}

    for zone_id, zone_data in zones_data.items():
        if zone_data is None:
            continue

        zone_name = zone_data['config']['name']
        print(f"Segmentando: {zone_name}")

        # Get data
        ndvi = ndvi_results[zone_id]
        seeds = seeds_by_zone[zone_id]

        # Initialize algorithm
        algorithm = ClassicRegionGrowing(
            threshold=threshold,
            min_region_size=min_region_size
        )

        # Execute segmentation
        start_time = time.time()
        labeled, num_regions, regions_info = algorithm.segment(ndvi, seeds)
        elapsed_time = time.time() - start_time

        # Calculate statistics
        coverage = np.sum(labeled > 0) / labeled.size * 100
        speed = labeled.size / elapsed_time / 1e6  # Million pixels per second

        # Save results
        segmentation_results[zone_id] = {
            'labeled': labeled,
            'num_regions': num_regions,
            'regions_info': regions_info,
            'elapsed_time': elapsed_time,
            'algorithm': algorithm
        }

        print(f"  ✓ Completado en {elapsed_time:.2f}s")
        print(f"    - Regiones: {num_regions}")
        print(f"    - Cobertura: {coverage:.1f}%")
        print(f"    - Velocidad: {speed:.2f} M píxeles/s")
        print()

    print("="*80)
    print(f"✓ SEGMENTACIÓN COMPLETADA PARA {len(segmentation_results)} ZONAS")
    print("="*80 + "\n")

    return segmentation_results


def run_mgrg_segmentation(
    zones_data: Dict,
    embeddings_data: Dict,
    threshold: float = 0.85,
    min_region_size: int = 50,
    seeds_by_zone: Optional[Dict] = None,
    segmentation_results: Optional[Dict] = None,
    zone_mapping: Optional[Dict] = None
) -> Dict:
    """
    Run MGRG (Metric-Guided Region Growing) segmentation on multiple zones.
    
    This function executes semantic region growing using embeddings from
    a foundation model (e.g., Prithvi) across multiple agricultural zones.
    
    Parameters
    ----------
    zones_data : dict
        Dictionary with zone configuration data
        Format: {zone_id: {'config': {'name': str, ...}, ...}}
    embeddings_data : dict
        Dictionary with embeddings for each zone
        Format: {zone_key: {'embeddings': np.ndarray}}
    threshold : float, default=0.85
        Cosine similarity threshold for region growing
    min_region_size : int, default=50
        Minimum number of pixels for a valid region
    seeds_by_zone : dict, optional
        Pre-computed seeds for each zone. If None, generates grid seeds.
        Format: {zone_id: [(y1, x1), (y2, x2), ...]}
    segmentation_results : dict, optional
        Classic RG results for comparison
        Format: {zone_id: {'num_regions': int, ...}}
    zone_mapping : dict, optional
        Mapping from embedding keys to zone IDs
        If None, assumes keys match zone IDs
        
    Returns
    -------
    dict
        Dictionary with MGRG results for each zone:
        {
            zone_id: {
                'labeled': np.ndarray,
                'num_regions': int,
                'regions_info': dict,
                'elapsed_time': float,
                'algorithm': SemanticRegionGrowing,
                'embeddings': np.ndarray
            }
        }
        
    Examples
    --------
    >>> from src.algorithms.semantic_region_growing import SemanticRegionGrowing
    >>> mgrg_results = run_mgrg_segmentation(
    ...     zones_data=zones_data,
    ...     embeddings_data=embeddings_data,
    ...     threshold=0.85,
    ...     min_region_size=50
    ... )
    >>> print(f"Processed {len(mgrg_results)} zones")
    """
    from src.algorithms.semantic_region_growing import SemanticRegionGrowing
    
    # Default zone mapping (assumes keys match)
    if zone_mapping is None:
        zone_mapping = {k: k for k in embeddings_data.keys()}
    
    print("\n" + "="*80)
    print("SEGMENTACIÓN CON MGRG (METRIC-GUIDED REGION GROWING)")
    print("="*80)
    print("\nParámetros:")
    print(f"  • Threshold (cosine similarity): {threshold}")
    print(f"  • Min region size: {min_region_size} píxeles")
    print(f"  • Criterio: Similitud semántica en espacio de embeddings")
    print("-"*80 + "\n")
    
    mgrg_results = {}
    
    for emb_key, zone_id in zone_mapping.items():
        # Verify embeddings exist
        if emb_key not in embeddings_data:
            print(f"⚠️ Embeddings no encontrados para {emb_key}")
            continue
        
        # Verify zone configuration exists
        if zone_id not in zones_data or zones_data[zone_id] is None:
            print(f"⚠️ Configuración no encontrada para {zone_id}")
            continue
        
        zone_display_name = zones_data[zone_id]['config']['name']
        print(f"Segmentando: {zone_display_name}")
        
        # Get embeddings
        embeddings = embeddings_data[emb_key]['embeddings']
        print(f"  Embeddings shape: {embeddings.shape}")
        
        # Get or generate seeds
        if seeds_by_zone and zone_id in seeds_by_zone:
            seeds = seeds_by_zone[zone_id]
            print(f"  Semillas (reutilizadas): {len(seeds)}")
        else:
            # Generate grid seeds
            h, w = embeddings.shape[0], embeddings.shape[1]
            seeds = []
            for y in range(20, h, 20):
                for x in range(20, w, 20):
                    seeds.append((y, x))
            print(f"  Semillas (generadas): {len(seeds)}")
        
        # Initialize MGRG
        mgrg = SemanticRegionGrowing(
            threshold=threshold,
            min_region_size=min_region_size
        )
        
        # Execute segmentation
        print("  Ejecutando MGRG...")
        start_time = time.time()
        labeled, num_regions, regions_info = mgrg.segment(embeddings, seeds)
        elapsed_time = time.time() - start_time
        
        # Calculate statistics
        coverage = np.sum(labeled > 0) / labeled.size * 100
        
        # Compare with Classic RG if available
        if segmentation_results and zone_id in segmentation_results:
            classic_regions = segmentation_results[zone_id]['num_regions']
            reduction = (1 - num_regions / classic_regions) * 100
            print(f"  ✓ Completado en {elapsed_time:.2f}s")
            print(f"    - Regiones MGRG: {num_regions}")
            print(f"    - Regiones Classic RG: {classic_regions}")
            print(f"    - Reducción: {reduction:.1f}%")
            print(f"    - Cobertura: {coverage:.1f}%")
        else:
            print(f"  ✓ Completado en {elapsed_time:.2f}s")
            print(f"    - Regiones: {num_regions}")
            print(f"    - Cobertura: {coverage:.1f}%")
        
        # Store results
        mgrg_results[zone_id] = {
            'labeled': labeled,
            'num_regions': num_regions,
            'regions_info': regions_info,
            'elapsed_time': elapsed_time,
            'algorithm': mgrg,
            'embeddings': embeddings
        }
        
        print()
    
    print("="*80)
    print(f"✓ MGRG COMPLETADO PARA {len(mgrg_results)} ZONAS")
    print("="*80 + "\n")
    
    return mgrg_results


def generate_grid_seeds(
    shape: Tuple[int, int],
    spacing: int = 20,
    offset: int = 20
) -> List[Tuple[int, int]]:
    """
    Generate grid-based seed points for region growing.
    
    Parameters
    ----------
    shape : tuple
        Shape of the image (height, width)
    spacing : int, default=20
        Spacing between seed points
    offset : int, default=20
        Offset from image borders
        
    Returns
    -------
    list
        List of seed coordinates [(y1, x1), (y2, x2), ...]
        
    Examples
    --------
    >>> seeds = generate_grid_seeds((512, 512), spacing=20)
    >>> print(f"Generated {len(seeds)} seeds")
    """
    h, w = shape
    seeds = []
    
    for y in range(offset, h, spacing):
        for x in range(offset, w, spacing):
            seeds.append((y, x))
    
    return seeds


def compare_segmentation_results(
    classic_results: Dict,
    mgrg_results: Dict,
    zones_data: Dict
) -> Dict:
    """
    Compare Classic RG and MGRG segmentation results.
    
    Parameters
    ----------
    classic_results : dict
        Classic Region Growing results
    mgrg_results : dict
        MGRG results
    zones_data : dict
        Zone configuration data
        
    Returns
    -------
    dict
        Comparison statistics for each zone
        
    Examples
    --------
    >>> comparison = compare_segmentation_results(
    ...     classic_results, mgrg_results, zones_data
    ... )
    >>> for zone_id, stats in comparison.items():
    ...     print(f"{zone_id}: {stats['reduction_pct']:.1f}% reduction")
    """
    comparison = {}
    
    for zone_id in classic_results.keys():
        if zone_id not in mgrg_results:
            continue
        
        zone_name = zones_data[zone_id]['config']['name']
        classic_regions = classic_results[zone_id]['num_regions']
        mgrg_regions = mgrg_results[zone_id]['num_regions']
        
        reduction_pct = (1 - mgrg_regions / classic_regions) * 100
        
        classic_coverage = (
            np.sum(classic_results[zone_id]['labeled'] > 0) /
            classic_results[zone_id]['labeled'].size * 100
        )
        mgrg_coverage = (
            np.sum(mgrg_results[zone_id]['labeled'] > 0) /
            mgrg_results[zone_id]['labeled'].size * 100
        )
        
        comparison[zone_id] = {
            'zone_name': zone_name,
            'classic_regions': classic_regions,
            'mgrg_regions': mgrg_regions,
            'reduction_pct': reduction_pct,
            'classic_coverage': classic_coverage,
            'mgrg_coverage': mgrg_coverage,
            'classic_time': classic_results[zone_id].get('elapsed_time', 0),
            'mgrg_time': mgrg_results[zone_id]['elapsed_time']
        }
    
    return comparison


def print_comparison_summary(comparison: Dict, return_df: bool = False, use_display: bool = True):
    """
    Print a formatted summary of segmentation comparison using pandas DataFrame.
    
    Parameters
    ----------
    comparison : dict
        Comparison statistics from compare_segmentation_results()
    return_df : bool, default=False
        If True, returns the DataFrame instead of just printing
    use_display : bool, default=True
        If True, uses IPython display for better formatting in Jupyter
        
    Returns
    -------
    pandas.DataFrame or None
        DataFrame with comparison results if return_df=True
        
    Examples
    --------
    >>> comparison = compare_segmentation_results(...)
    >>> print_comparison_summary(comparison)
    >>> # Or get the DataFrame
    >>> df = print_comparison_summary(comparison, return_df=True)
    """
    import pandas as pd
    
    # Prepare data for DataFrame
    data = []
    for zone_id, stats in comparison.items():
        data.append({
            'Zona': stats['zone_name'],
            'Classic RG Regiones': stats['classic_regions'],
            'Classic RG Cobertura (%)': f"{stats['classic_coverage']:.1f}",
            'Classic RG Tiempo (s)': f"{stats['classic_time']:.2f}",
            'MGRG Regiones': stats['mgrg_regions'],
            'MGRG Cobertura (%)': f"{stats['mgrg_coverage']:.1f}",
            'MGRG Tiempo (s)': f"{stats['mgrg_time']:.2f}",
            'Reducción (%)': f"{stats['reduction_pct']:.1f}"
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Calculate averages
    avg_reduction = np.mean([s['reduction_pct'] for s in comparison.values()])
    avg_classic_regions = np.mean([s['classic_regions'] for s in comparison.values()])
    avg_mgrg_regions = np.mean([s['mgrg_regions'] for s in comparison.values()])
    
    print("\n" + "="*100)
    print("COMPARACIÓN: CLASSIC RG vs MGRG")
    print("="*100 + "\n")
    
    # Display DataFrame
    if use_display:
        try:
            from IPython.display import display
            display(df)
        except ImportError:
            print(df.to_string(index=False))
    else:
        print(df.to_string(index=False))
    
    # Print summary statistics
    print("\n" + "-"*100)
    print("ESTADÍSTICAS GENERALES:")
    print("-"*100)
    print(f"  Promedio de regiones Classic RG: {avg_classic_regions:.0f}")
    print(f"  Promedio de regiones MGRG:       {avg_mgrg_regions:.0f}")
    print(f"  Reducción promedio:              {avg_reduction:.1f}%")
    print("="*100 + "\n")
    
    if return_df:
        return df
    return None


def create_detailed_comparison_df(comparison: Dict):
    """
    Create a detailed comparison DataFrame with better formatting.
    
    Parameters
    ----------
    comparison : dict
        Comparison statistics from compare_segmentation_results()
        
    Returns
    -------
    pandas.DataFrame
        Detailed comparison DataFrame
        
    Examples
    --------
    >>> comparison = compare_segmentation_results(...)
    >>> df = create_detailed_comparison_df(comparison)
    >>> display(df)  # In Jupyter
    """
    import pandas as pd
    
    # Create separate DataFrames for each metric
    zones = [stats['zone_name'] for stats in comparison.values()]
    
    # Regions comparison
    regions_data = {
        'Zona': zones,
        'Classic RG': [s['classic_regions'] for s in comparison.values()],
        'MGRG': [s['mgrg_regions'] for s in comparison.values()],
        'Reducción (%)': [f"{s['reduction_pct']:.1f}" for s in comparison.values()]
    }
    df_regions = pd.DataFrame(regions_data)
    
    # Coverage comparison
    coverage_data = {
        'Zona': zones,
        'Classic RG (%)': [f"{s['classic_coverage']:.1f}" for s in comparison.values()],
        'MGRG (%)': [f"{s['mgrg_coverage']:.1f}" for s in comparison.values()],
        'Diferencia': [f"{s['mgrg_coverage'] - s['classic_coverage']:.1f}" for s in comparison.values()]
    }
    df_coverage = pd.DataFrame(coverage_data)
    
    # Time comparison
    time_data = {
        'Zona': zones,
        'Classic RG (s)': [f"{s['classic_time']:.2f}" for s in comparison.values()],
        'MGRG (s)': [f"{s['mgrg_time']:.2f}" for s in comparison.values()],
        'Diferencia (s)': [f"{s['mgrg_time'] - s['classic_time']:.2f}" for s in comparison.values()]
    }
    df_time = pd.DataFrame(time_data)
    
    return {
        'regions': df_regions,
        'coverage': df_coverage,
        'time': df_time
    }


def display_comparison_tables(comparison: Dict, use_display: bool = True) -> None:
    """
    Display multiple comparison tables with nice formatting.
    
    Parameters
    ----------
    comparison : dict
        Comparison statistics from compare_segmentation_results()
    use_display : bool, default=True
        If True, uses IPython display for better formatting in Jupyter
        
    Examples
    --------
    >>> comparison = compare_segmentation_results(...)
    >>> display_comparison_tables(comparison)
    """
    dfs = create_detailed_comparison_df(comparison)
    
    try:
        from IPython.display import display, HTML
        
        if use_display:
            print("\n" + "="*80)
            print("COMPARACIÓN DE NÚMERO DE REGIONES")
            print("="*80)
            display(dfs['regions'])
            
            print("\n" + "="*80)
            print("COMPARACIÓN DE COBERTURA")
            print("="*80)
            display(dfs['coverage'])
            
            print("\n" + "="*80)
            print("COMPARACIÓN DE TIEMPO DE EJECUCIÓN")
            print("="*80)
            display(dfs['time'])
            print("="*80 + "\n")
        else:
            raise ImportError  # Fall back to print
    except ImportError:
        print("\n" + "="*80)
        print("COMPARACIÓN DE NÚMERO DE REGIONES")
        print("="*80)
        print(dfs['regions'].to_string(index=False))
        
        print("\n" + "="*80)
        print("COMPARACIÓN DE COBERTURA")
        print("="*80)
        print(dfs['coverage'].to_string(index=False))
        
        print("\n" + "="*80)
        print("COMPARACIÓN DE TIEMPO DE EJECUCIÓN")
        print("="*80)
        print(dfs['time'].to_string(index=False))
        print("="*80 + "\n")



def classify_stress_levels(
    segmentation_results: Dict,
    ndvi_results: Dict,
    zones_data: Dict,
    thresholds: Dict = None,
    use_display: bool = True
) -> Dict:
    """
    Classify regions by vegetation stress level based on NDVI values.
    
    This function works with both Classic RG and MGRG segmentation results.
    
    Parameters
    ----------
    segmentation_results : dict
        Segmentation results (Classic RG or MGRG)
        Format: {zone_id: {'labeled': array, 'regions_info': list}}
    ndvi_results : dict
        NDVI arrays for each zone
        Format: {zone_id: ndvi_array}
    zones_data : dict
        Zone configuration data
    thresholds : dict, optional
        Custom NDVI thresholds. Default: {'high': 0.3, 'medium': 0.5}
    use_display : bool, default=True
        If True, uses IPython display for DataFrame output
        
    Returns
    -------
    dict
        Stress classification results:
        {
            zone_id: {
                'classified': {
                    'high_stress': [...],
                    'medium_stress': [...],
                    'low_stress': [...]
                },
                'stress_counts': {'high': int, 'medium': int, 'low': int}
            }
        }
        
    Examples
    --------
    >>> stress_results = classify_stress_levels(
    ...     segmentation_results=mgrg_results,
    ...     ndvi_results=ndvi_results,
    ...     zones_data=zones_data
    ... )
    """
    import pandas as pd
    
    # Default thresholds
    if thresholds is None:
        thresholds = {'high': 0.3, 'medium': 0.5}
    
    print("\n" + "="*80)
    print("CLASIFICACIÓN POR NIVEL DE ESTRÉS")
    print("="*80)
    print("\nCriterios (basados en NDVI):")
    print(f"  • Alto:   NDVI < {thresholds['high']}")
    print(f"  • Medio:  {thresholds['high']} ≤ NDVI < {thresholds['medium']}")
    print(f"  • Bajo:   NDVI ≥ {thresholds['medium']}")
    print("-"*80 + "\n")
    
    stress_classification = {}
    summary_data = []
    
    for zone_id, seg_result in segmentation_results.items():
        if zone_id not in ndvi_results:
            print(f"⚠️ NDVI no encontrado para zona {zone_id}")
            continue
        
        zone_name = zones_data[zone_id]['config']['name']
        print(f"Clasificando: {zone_name}")
        
        # Get data
        labeled = seg_result['labeled']
        ndvi = ndvi_results[zone_id]
        regions_info = seg_result.get('regions_info', [])
        
        # If regions_info is empty, create it from labeled array
        if not regions_info:
            unique_regions = np.unique(labeled)
            unique_regions = unique_regions[unique_regions > 0]  # Exclude background
            regions_info = [{'id': int(rid)} for rid in unique_regions]
        
        # Classify by stress using NDVI
        classified_stress = {
            'high_stress': [],
            'medium_stress': [],
            'low_stress': []
        }
        
        for region in regions_info:
            region_id = region['id']
            mask = (labeled == region_id)
            region_ndvi = ndvi[mask]
            
            if region_ndvi.size == 0:
                continue
            
            mean_ndvi = np.mean(region_ndvi)
            
            # Add mean_ndvi to region info
            region['mean_ndvi'] = float(mean_ndvi)
            
            # Classify
            if mean_ndvi < thresholds['high']:
                region['stress_level'] = 'high'
                classified_stress['high_stress'].append(region)
            elif mean_ndvi < thresholds['medium']:
                region['stress_level'] = 'medium'
                classified_stress['medium_stress'].append(region)
            else:
                region['stress_level'] = 'low'
                classified_stress['low_stress'].append(region)
        
        # Count by level
        stress_counts = {
            'high': len(classified_stress['high_stress']),
            'medium': len(classified_stress['medium_stress']),
            'low': len(classified_stress['low_stress'])
        }
        
        stress_classification[zone_id] = {
            'classified': classified_stress,
            'stress_counts': stress_counts
        }
        
        print("  ✓ Clasificación completada")
        print(f"    - Estrés alto:   {stress_counts['high']} regiones")
        print(f"    - Estrés medio:  {stress_counts['medium']} regiones")
        print(f"    - Estrés bajo:   {stress_counts['low']} regiones\n")
        
        # Collect data for summary table
        total_regions = sum(stress_counts.values())
        summary_data.append({
            'Zona': zone_name,
            'Total Regiones': total_regions,
            'Alto': stress_counts['high'],
            'Alto (%)': f"{stress_counts['high']/total_regions*100:.1f}" if total_regions > 0 else "0.0",
            'Medio': stress_counts['medium'],
            'Medio (%)': f"{stress_counts['medium']/total_regions*100:.1f}" if total_regions > 0 else "0.0",
            'Bajo': stress_counts['low'],
            'Bajo (%)': f"{stress_counts['low']/total_regions*100:.1f}" if total_regions > 0 else "0.0"
        })
    
    # Display summary table
    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        
        print("="*80)
        print("RESUMEN DE CLASIFICACIÓN POR ESTRÉS")
        print("="*80 + "\n")
        
        if use_display:
            try:
                from IPython.display import display
                display(df_summary)
            except ImportError:
                print(df_summary.to_string(index=False))
        else:
            print(df_summary.to_string(index=False))
        
        print("\n" + "="*80)
    
    print(f"✓ Clasificación por estrés completada para {len(stress_classification)} zonas\n")
    
    return stress_classification


def get_stress_summary_df(stress_classification: Dict, zones_data: Dict):
    """
    Create a summary DataFrame of stress classification results.
    
    Parameters
    ----------
    stress_classification : dict
        Results from classify_stress_levels()
    zones_data : dict
        Zone configuration data
        
    Returns
    -------
    pandas.DataFrame
        Summary DataFrame with stress statistics
        
    Examples
    --------
    >>> df = get_stress_summary_df(stress_results, zones_data)
    >>> display(df)
    """
    import pandas as pd
    
    data = []
    for zone_id, results in stress_classification.items():
        zone_name = zones_data[zone_id]['config']['name']
        counts = results['stress_counts']
        total = sum(counts.values())
        
        data.append({
            'Zona': zone_name,
            'Total Regiones': total,
            'Estrés Alto': counts['high'],
            'Estrés Alto (%)': f"{counts['high']/total*100:.1f}" if total > 0 else "0.0",
            'Estrés Medio': counts['medium'],
            'Estrés Medio (%)': f"{counts['medium']/total*100:.1f}" if total > 0 else "0.0",
            'Estrés Bajo': counts['low'],
            'Estrés Bajo (%)': f"{counts['low']/total*100:.1f}" if total > 0 else "0.0"
        })
    
    return pd.DataFrame(data)



def calculate_quality_metrics(
    segmentation_results: Dict,
    ndvi_results: Dict,
    zones_data: Dict,
    pixel_size_m: float = 10.0,
    use_display: bool = True
):
    """
    Calculate quality metrics for segmentation results.
    
    Calculates spatial coherence, region sizes, and area statistics
    for each zone and returns a formatted DataFrame.
    
    Parameters
    ----------
    segmentation_results : dict
        Segmentation results (Classic RG or MGRG)
        Format: {zone_id: {'labeled': array, 'num_regions': int, 'regions_info': list}}
    ndvi_results : dict
        NDVI arrays for each zone
    zones_data : dict
        Zone configuration data
    pixel_size_m : float, default=10.0
        Pixel size in meters (for Sentinel-2: 10m)
    use_display : bool, default=True
        If True, uses IPython display for DataFrame output
        
    Returns
    -------
    tuple
        (metrics_dict, metrics_df) where:
        - metrics_dict: Dictionary with detailed metrics per zone
        - metrics_df: pandas DataFrame with formatted metrics
        
    Examples
    --------
    >>> metrics, df = calculate_quality_metrics(
    ...     segmentation_results=mgrg_results,
    ...     ndvi_results=ndvi_results,
    ...     zones_data=zones_data
    ... )
    >>> display(df)
    """
    import pandas as pd
    
    quality_metrics = {}
    df_data = []
    
    for zone_id, seg_result in segmentation_results.items():
        if zone_id not in ndvi_results:
            continue
        
        zone_name = zones_data[zone_id]['config']['name']
        labeled = seg_result['labeled']
        num_regions = seg_result['num_regions']
        ndvi = ndvi_results[zone_id]
        regions_info = seg_result.get('regions_info', [])
        
        # 1. Spatial coherence (based on NDVI homogeneity within regions)
        coherence_scores = []
        for region_id in range(1, num_regions + 1):
            mask = (labeled == region_id)
            if np.sum(mask) > 0:
                region_ndvi = ndvi[mask]
                std = np.std(region_ndvi)
                coherence = 1.0 - std  # Higher coherence = lower std
                coherence_scores.append(coherence)
        
        avg_coherence = np.mean(coherence_scores) * 100 if coherence_scores else 0
        
        # 2. Region sizes
        if regions_info:
            region_sizes = [info['size'] for info in regions_info]
        else:
            # Calculate from labeled array if regions_info not available
            region_sizes = []
            for region_id in range(1, num_regions + 1):
                size = np.sum(labeled == region_id)
                if size > 0:
                    region_sizes.append(size)
        
        avg_size = np.mean(region_sizes) if region_sizes else 0
        min_size = min(region_sizes) if region_sizes else 0
        max_size = max(region_sizes) if region_sizes else 0
        std_size = np.std(region_sizes) if region_sizes else 0
        
        # 3. Area in hectares
        pixel_area_m2 = pixel_size_m * pixel_size_m
        avg_area_ha = (avg_size * pixel_area_m2) / 10000
        min_area_ha = (min_size * pixel_area_m2) / 10000
        max_area_ha = (max_size * pixel_area_m2) / 10000
        
        # 4. Coverage
        coverage = np.sum(labeled > 0) / labeled.size * 100
        
        # Store detailed metrics
        quality_metrics[zone_id] = {
            'coherence': avg_coherence,
            'avg_size': avg_size,
            'min_size': min_size,
            'max_size': max_size,
            'std_size': std_size,
            'avg_area_ha': avg_area_ha,
            'min_area_ha': min_area_ha,
            'max_area_ha': max_area_ha,
            'coverage': coverage,
            'num_regions': num_regions
        }
        
        # Prepare data for DataFrame
        df_data.append({
            'Zona': zone_name,
            'Regiones': num_regions,
            'Coherencia (%)': f"{avg_coherence:.1f}",
            'Cobertura (%)': f"{coverage:.1f}",
            'Tamaño Promedio (px)': f"{avg_size:.0f}",
            'Tamaño Promedio (ha)': f"{avg_area_ha:.2f}",
            'Rango Tamaño (px)': f"{min_size} - {max_size}",
            'Desv. Est. Tamaño': f"{std_size:.0f}"
        })
    
    # Create DataFrame
    df_metrics = pd.DataFrame(df_data)
    
    # Display
    print("\n" + "="*100)
    print("MÉTRICAS DE CALIDAD DE SEGMENTACIÓN")
    print("="*100 + "\n")
    
    if use_display:
        try:
            from IPython.display import display
            display(df_metrics)
        except ImportError:
            print(df_metrics.to_string(index=False))
    else:
        print(df_metrics.to_string(index=False))
    
    print("\n" + "="*100 + "\n")
    
    return quality_metrics, df_metrics


def compare_quality_metrics(
    classic_metrics: Dict,
    mgrg_metrics: Dict,
    zones_data: Dict,
    use_display: bool = True
):
    """
    Compare quality metrics between Classic RG and MGRG.
    
    Parameters
    ----------
    classic_metrics : dict
        Quality metrics from Classic RG
    mgrg_metrics : dict
        Quality metrics from MGRG
    zones_data : dict
        Zone configuration data
    use_display : bool, default=True
        If True, uses IPython display for DataFrame output
        
    Returns
    -------
    pandas.DataFrame
        Comparison DataFrame
        
    Examples
    --------
    >>> classic_m, _ = calculate_quality_metrics(segmentation_results, ...)
    >>> mgrg_m, _ = calculate_quality_metrics(mgrg_results, ...)
    >>> df_comp = compare_quality_metrics(classic_m, mgrg_m, zones_data)
    """
    import pandas as pd
    
    data = []
    for zone_id in classic_metrics.keys():
        if zone_id not in mgrg_metrics:
            continue
        
        zone_name = zones_data[zone_id]['config']['name']
        classic = classic_metrics[zone_id]
        mgrg = mgrg_metrics[zone_id]
        
        # Calculate improvements
        coherence_improvement = mgrg['coherence'] - classic['coherence']
        size_ratio = mgrg['avg_size'] / classic['avg_size'] if classic['avg_size'] > 0 else 0
        
        data.append({
            'Zona': zone_name,
            'Classic RG Coherencia (%)': f"{classic['coherence']:.1f}",
            'MGRG Coherencia (%)': f"{mgrg['coherence']:.1f}",
            'Mejora Coherencia': f"{coherence_improvement:+.1f}",
            'Classic RG Tamaño (ha)': f"{classic['avg_area_ha']:.2f}",
            'MGRG Tamaño (ha)': f"{mgrg['avg_area_ha']:.2f}",
            'Ratio Tamaño': f"{size_ratio:.1f}x",
            'Classic RG Regiones': classic['num_regions'],
            'MGRG Regiones': mgrg['num_regions']
        })
    
    df_comparison = pd.DataFrame(data)
    
    print("\n" + "="*120)
    print("COMPARACIÓN DE MÉTRICAS DE CALIDAD: CLASSIC RG vs MGRG")
    print("="*120 + "\n")
    
    if use_display:
        try:
            from IPython.display import display
            display(df_comparison)
        except ImportError:
            print(df_comparison.to_string(index=False))
    else:
        print(df_comparison.to_string(index=False))
    
    print("\n" + "="*120 + "\n")
    
    return df_comparison



def create_comprehensive_summary(
    segmentation_results: Dict,
    quality_metrics: Dict,
    stress_classification: Dict,
    zones_data: Dict,
    method_name: str = "MGRG",
    use_display: bool = True
):
    """
    Create a comprehensive summary comparing all zones.
    
    Combines segmentation results, quality metrics, and stress classification
    into a single summary DataFrame with averages and insights.
    
    Parameters
    ----------
    segmentation_results : dict
        Segmentation results (Classic RG or MGRG)
    quality_metrics : dict
        Quality metrics from calculate_quality_metrics()
    stress_classification : dict
        Stress classification from classify_stress_levels()
    zones_data : dict
        Zone configuration data
    method_name : str, default="MGRG"
        Name of the segmentation method for display
    use_display : bool, default=True
        If True, uses IPython display for DataFrame output
        
    Returns
    -------
    tuple
        (summary_df, insights_dict) where:
        - summary_df: pandas DataFrame with complete summary
        - insights_dict: Dictionary with key insights
        
    Examples
    --------
    >>> summary_df, insights = create_comprehensive_summary(
    ...     segmentation_results=mgrg_results,
    ...     quality_metrics=mgrg_quality_metrics,
    ...     stress_classification=mgrg_stress_classification,
    ...     zones_data=zones_data,
    ...     method_name="MGRG"
    ... )
    """
    import pandas as pd
    
    summary_data = []
    
    for zone_id in segmentation_results.keys():
        if zone_id not in quality_metrics or zone_id not in stress_classification:
            continue
        
        zone_name = zones_data[zone_id]['config']['name']
        
        # Get data
        seg = segmentation_results[zone_id]
        quality = quality_metrics[zone_id]
        stress = stress_classification[zone_id]
        
        summary_data.append({
            'Zona': zone_name,
            'Regiones': seg['num_regions'],
            'Coherencia (%)': f"{quality['coherence']:.1f}",
            'Cobertura (%)': f"{quality['coverage']:.1f}",
            'Tiempo (s)': f"{seg.get('elapsed_time', 0):.2f}",
            'Tamaño Prom (px)': f"{quality['avg_size']:.0f}",
            'Tamaño Prom (ha)': f"{quality['avg_area_ha']:.2f}",
            'Estrés Alto': stress['stress_counts']['high'],
            'Estrés Medio': stress['stress_counts']['medium'],
            'Estrés Bajo': stress['stress_counts']['low']
        })
    
    df_summary = pd.DataFrame(summary_data)
    
    # Calculate averages
    avg_row = {
        'Zona': 'PROMEDIO',
        'Regiones': int(df_summary['Regiones'].mean()),
        'Coherencia (%)': f"{df_summary['Coherencia (%)'].str.rstrip('%').astype(float).mean():.1f}",
        'Cobertura (%)': f"{df_summary['Cobertura (%)'].str.rstrip('%').astype(float).mean():.1f}",
        'Tiempo (s)': f"{df_summary['Tiempo (s)'].astype(float).mean():.2f}",
        'Tamaño Prom (px)': f"{df_summary['Tamaño Prom (px)'].astype(float).mean():.0f}",
        'Tamaño Prom (ha)': f"{df_summary['Tamaño Prom (ha)'].astype(float).mean():.2f}",
        'Estrés Alto': int(df_summary['Estrés Alto'].mean()),
        'Estrés Medio': int(df_summary['Estrés Medio'].mean()),
        'Estrés Bajo': int(df_summary['Estrés Bajo'].mean())
    }
    
    # Add average row
    df_summary = pd.concat([df_summary, pd.DataFrame([avg_row])], ignore_index=True)
    
    # Create insights dictionary
    insights = {
        'avg_coherence': float(avg_row['Coherencia (%)'].rstrip('%')),
        'avg_coverage': float(avg_row['Cobertura (%)'].rstrip('%')),
        'avg_regions': avg_row['Regiones'],
        'avg_area_ha': float(avg_row['Tamaño Prom (ha)']),
        'avg_time': float(avg_row['Tiempo (s)']),
        'total_high_stress': df_summary['Estrés Alto'].iloc[:-1].sum(),
        'total_medium_stress': df_summary['Estrés Medio'].iloc[:-1].sum(),
        'total_low_stress': df_summary['Estrés Bajo'].iloc[:-1].sum()
    }
    
    # Display
    print("\n" + "="*120)
    print(f"RESUMEN COMPARATIVO ENTRE ZONAS - {method_name}")
    print("="*120 + "\n")
    
    if use_display:
        try:
            from IPython.display import display
            display(df_summary)
        except ImportError:
            print(df_summary.to_string(index=False))
    else:
        print(df_summary.to_string(index=False))
    
    # Display insights
    print("\n" + "-"*120)
    print("INSIGHTS CLAVE:")
    print("-"*120)
    print(f"  • Coherencia espacial promedio:      {insights['avg_coherence']:.1f}%")
    print(f"  • Cobertura promedio:                 {insights['avg_coverage']:.1f}%")
    print(f"  • Regiones promedio por zona:         {insights['avg_regions']}")
    print(f"  • Tamaño promedio de región:          {insights['avg_area_ha']:.2f} hectáreas")
    print(f"  • Tiempo promedio de procesamiento:   {insights['avg_time']:.2f} segundos")
    print(f"\n  • Total regiones con estrés alto:     {insights['total_high_stress']}")
    print(f"  • Total regiones con estrés medio:    {insights['total_medium_stress']}")
    print(f"  • Total regiones con estrés bajo:     {insights['total_low_stress']}")
    print("="*120 + "\n")
    
    return df_summary, insights


def create_method_comparison_summary(
    classic_results: Dict,
    classic_quality: Dict,
    classic_stress: Dict,
    mgrg_results: Dict,
    mgrg_quality: Dict,
    mgrg_stress: Dict,
    zones_data: Dict,
    use_display: bool = True
):
    """
    Create a side-by-side comparison summary of Classic RG and MGRG.
    
    Parameters
    ----------
    classic_results : dict
        Classic RG segmentation results
    classic_quality : dict
        Classic RG quality metrics
    classic_stress : dict
        Classic RG stress classification
    mgrg_results : dict
        MGRG segmentation results
    mgrg_quality : dict
        MGRG quality metrics
    mgrg_stress : dict
        MGRG stress classification
    zones_data : dict
        Zone configuration data
    use_display : bool, default=True
        If True, uses IPython display for DataFrame output
        
    Returns
    -------
    pandas.DataFrame
        Comparison summary DataFrame
        
    Examples
    --------
    >>> df_comp = create_method_comparison_summary(
    ...     classic_results, classic_quality, classic_stress,
    ...     mgrg_results, mgrg_quality, mgrg_stress,
    ...     zones_data
    ... )
    """
    import pandas as pd
    
    data = []
    
    for zone_id in classic_results.keys():
        if zone_id not in mgrg_results:
            continue
        
        zone_name = zones_data[zone_id]['config']['name']
        
        # Classic RG data
        c_seg = classic_results[zone_id]
        c_qual = classic_quality[zone_id]
        c_stress = classic_stress[zone_id]
        
        # MGRG data
        m_seg = mgrg_results[zone_id]
        m_qual = mgrg_quality[zone_id]
        m_stress = mgrg_stress[zone_id]
        
        # Calculate improvements
        region_reduction = (1 - m_seg['num_regions'] / c_seg['num_regions']) * 100
        coherence_improvement = m_qual['coherence'] - c_qual['coherence']
        
        data.append({
            'Zona': zone_name,
            'Classic Regiones': c_seg['num_regions'],
            'MGRG Regiones': m_seg['num_regions'],
            'Reducción (%)': f"{region_reduction:.1f}",
            'Classic Coherencia': f"{c_qual['coherence']:.1f}%",
            'MGRG Coherencia': f"{m_qual['coherence']:.1f}%",
            'Mejora': f"{coherence_improvement:+.1f}%",
            'Classic Estrés Alto': c_stress['stress_counts']['high'],
            'MGRG Estrés Alto': m_stress['stress_counts']['high']
        })
    
    df_comparison = pd.DataFrame(data)
    
    # Calculate averages
    avg_reduction = df_comparison['Reducción (%)'].str.rstrip('%').astype(float).mean()
    avg_classic_coherence = df_comparison['Classic Coherencia'].str.rstrip('%').astype(float).mean()
    avg_mgrg_coherence = df_comparison['MGRG Coherencia'].str.rstrip('%').astype(float).mean()
    
    print("\n" + "="*120)
    print("COMPARACIÓN GENERAL: CLASSIC RG vs MGRG")
    print("="*120 + "\n")
    
    if use_display:
        try:
            from IPython.display import display
            display(df_comparison)
        except ImportError:
            print(df_comparison.to_string(index=False))
    else:
        print(df_comparison.to_string(index=False))
    
    print("\n" + "-"*120)
    print("RESUMEN DE MEJORAS:")
    print("-"*120)
    print(f"  • Reducción promedio de regiones:     {avg_reduction:.1f}%")
    print(f"  • Coherencia Classic RG:              {avg_classic_coherence:.1f}%")
    print(f"  • Coherencia MGRG:                    {avg_mgrg_coherence:.1f}%")
    print(f"  • Mejora en coherencia:               {avg_mgrg_coherence - avg_classic_coherence:+.1f}%")
    print("="*120 + "\n")
    
    return df_comparison



def create_detailed_method_comparison(
    classic_results: Dict,
    classic_quality: Dict,
    classic_stress: Dict,
    mgrg_results: Dict,
    mgrg_quality: Dict,
    mgrg_stress: Dict,
    zones_data: Dict,
    use_display: bool = True
):
    """
    Create a detailed side-by-side comparison with improvement rows.
    
    Creates a DataFrame with alternating rows showing Classic RG, MGRG,
    and improvement metrics for each zone.
    
    Parameters
    ----------
    classic_results : dict
        Classic RG segmentation results
    classic_quality : dict
        Classic RG quality metrics
    classic_stress : dict
        Classic RG stress classification
    mgrg_results : dict
        MGRG segmentation results
    mgrg_quality : dict
        MGRG quality metrics
    mgrg_stress : dict
        MGRG stress classification
    zones_data : dict
        Zone configuration data
    use_display : bool, default=True
        If True, uses IPython display for DataFrame output
        
    Returns
    -------
    pandas.DataFrame
        Detailed comparison DataFrame with improvement rows
        
    Examples
    --------
    >>> df = create_detailed_method_comparison(
    ...     classic_results, classic_quality, classic_stress,
    ...     mgrg_results, mgrg_quality, mgrg_stress,
    ...     zones_data
    ... )
    """
    import pandas as pd
    
    comparison_data = []
    
    for zone_id in zones_data.keys():
        if zone_id not in classic_results or zone_id not in mgrg_results:
            continue
        
        zone_name = zones_data[zone_id]['config']['name']
        
        # Classic RG data
        classic = classic_results[zone_id]
        c_quality = classic_quality[zone_id]
        c_stress = classic_stress[zone_id]
        
        # MGRG data
        mgrg = mgrg_results[zone_id]
        m_quality = mgrg_quality[zone_id]
        m_stress = mgrg_stress[zone_id]
        
        # Calculate improvements
        region_reduction = (1 - mgrg['num_regions'] / classic['num_regions']) * 100
        coherence_improvement = m_quality['coherence'] - c_quality['coherence']
        
        # Classic RG row
        comparison_data.append({
            'Zona': zone_name,
            'Método': 'Classic RG',
            'Regiones': classic['num_regions'],
            'Coherencia (%)': f"{c_quality['coherence']:.1f}",
            'Tiempo (s)': f"{classic.get('elapsed_time', 0):.2f}",
            'Tamaño Prom (ha)': f"{c_quality['avg_area_ha']:.2f}",
            'Alto': c_stress['stress_counts']['high'],
            'Medio': c_stress['stress_counts']['medium'],
            'Bajo': c_stress['stress_counts']['low']
        })
        
        # MGRG row
        comparison_data.append({
            'Zona': '',
            'Método': 'MGRG',
            'Regiones': mgrg['num_regions'],
            'Coherencia (%)': f"{m_quality['coherence']:.1f}",
            'Tiempo (s)': f"{mgrg.get('elapsed_time', 0):.2f}",
            'Tamaño Prom (ha)': f"{m_quality['avg_area_ha']:.2f}",
            'Alto': m_stress['stress_counts']['high'],
            'Medio': m_stress['stress_counts']['medium'],
            'Bajo': m_stress['stress_counts']['low']
        })
        
        # Improvement row
        comparison_data.append({
            'Zona': '',
            'Método': '→ Mejora',
            'Regiones': f"{region_reduction:.1f}%",
            'Coherencia (%)': f"{coherence_improvement:+.1f}",
            'Tiempo (s)': '-',
            'Tamaño Prom (ha)': '-',
            'Alto': '-',
            'Medio': '-',
            'Bajo': '-'
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Display
    print("\n" + "="*120)
    print("RESUMEN COMPARATIVO DETALLADO: CLASSIC RG vs MGRG")
    print("="*120 + "\n")
    
    if use_display:
        try:
            from IPython.display import display
            display(df_comparison)
        except ImportError:
            print(df_comparison.to_string(index=False))
    else:
        print(df_comparison.to_string(index=False))
    
    # Calculate and display overall statistics
    classic_total_regions = sum([classic_results[zid]['num_regions'] 
                                  for zid in classic_results.keys() 
                                  if zid in mgrg_results])
    mgrg_total_regions = sum([mgrg_results[zid]['num_regions'] 
                               for zid in mgrg_results.keys() 
                               if zid in classic_results])
    overall_reduction = (1 - mgrg_total_regions / classic_total_regions) * 100
    
    avg_classic_coherence = np.mean([classic_quality[zid]['coherence'] 
                                      for zid in classic_quality.keys() 
                                      if zid in mgrg_quality])
    avg_mgrg_coherence = np.mean([mgrg_quality[zid]['coherence'] 
                                   for zid in mgrg_quality.keys() 
                                   if zid in classic_quality])
    
    print("\n" + "-"*120)
    print("ESTADÍSTICAS GENERALES:")
    print("-"*120)
    print(f"  • Reducción total de regiones:        {overall_reduction:.1f}%")
    print(f"  • Coherencia promedio Classic RG:     {avg_classic_coherence:.1f}%")
    print(f"  • Coherencia promedio MGRG:           {avg_mgrg_coherence:.1f}%")
    print(f"  • Mejora en coherencia:               {avg_mgrg_coherence - avg_classic_coherence:+.1f}%")
    print("="*120 + "\n")
    
    return df_comparison
