"""
Classification utilities for semantic analysis.

This module provides high-level functions for semantic classification
of segmented regions using SemanticClassifier.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from pathlib import Path
import logging

from src.classification.zero_shot_classifier import SemanticClassifier, LAND_COVER_CLASSES

logger = logging.getLogger(__name__)


def classify_all_zones(
    mgrg_results: Dict,
    ndvi_results: Dict,
    embeddings_data: Dict,
    zones_data: Dict,
    zone_mapping: Dict,
    resolution: float = 10.0,
    min_size: int = 50,
    return_dataframe: bool = True
) -> tuple:
    """
    Perform semantic classification for all zones.
    
    This function:
    1. Initializes SemanticClassifier for each zone
    2. Classifies all regions in each zone
    3. Generates semantic maps
    4. Calculates class statistics
    5. Creates a summary DataFrame
    
    Parameters
    ----------
    mgrg_results : dict
        MGRG segmentation results for each zone
        Format: {zone_id: {'labeled': array, ...}}
    ndvi_results : dict
        NDVI arrays for each zone
        Format: {zone_id: ndvi_array}
    embeddings_data : dict
        Embeddings data for each zone
        Format: {zone_name: {'embeddings': array, ...}}
    zones_data : dict
        Zone configuration data
        Format: {zone_id: {'config': {'name': str, ...}, ...}}
    zone_mapping : dict
        Mapping from zone keys to zone IDs
        Format: {zone_name: zone_id}
    resolution : float, default=10.0
        Spatial resolution in meters (for area calculation)
    min_size : int, default=50
        Skip regions smaller than this (pixels)
    return_dataframe : bool, default=True
        Whether to return formatted DataFrame
        
    Returns
    -------
    tuple
        (semantic_results, df_all_classes, df_all_classes_formatted)
        - semantic_results: dict with classification results per zone
        - df_all_classes: pandas DataFrame with raw data
        - df_all_classes_formatted: pandas DataFrame with formatted strings
        
    Examples
    --------
    >>> semantic_results, df_raw, df_formatted = classify_all_zones(
    ...     mgrg_results=mgrg_results,
    ...     ndvi_results=ndvi_results,
    ...     embeddings_data=embeddings_data,
    ...     zones_data=zones_data,
    ...     zone_mapping=zone_mapping
    ... )
    >>> print(f"Classified {len(semantic_results)} zones")
    >>> display(df_formatted)
    """
    print("CLASIFICACIÓN SEMÁNTICA ZERO-SHOT\n")
    
    semantic_results = {}
    all_class_data = []
    
    for zone_name, zone_id in zone_mapping.items():
        if zone_id not in mgrg_results:
            logger.warning(f"Zone {zone_name} not found in MGRG results, skipping")
            continue
        
        # Get data
        segmentation = mgrg_results[zone_id]['labeled']
        ndvi = ndvi_results[zone_id]
        embeddings = embeddings_data[zone_name]['embeddings']
        
        logger.info(f"Classifying zone: {zone_name}")
        
        # Initialize classifier
        classifier = SemanticClassifier(
            embeddings=embeddings,
            ndvi=ndvi,
            resolution=resolution
        )
        
        # Classify all regions
        classifications = classifier.classify_all_regions(
            segmentation=segmentation,
            min_size=min_size
        )
        
        # Generate semantic map
        semantic_map = classifier.generate_semantic_map(
            segmentation=segmentation,
            classifications=classifications
        )
        
        # Get statistics per class
        class_stats = classifier.get_class_statistics(classifications)
        
        # Save results
        semantic_results[zone_id] = {
            'semantic_map': semantic_map,
            'classifications': classifications,
            'class_stats': class_stats
        }
        
        # Prepare data for DataFrame
        zone_display = zones_data[zone_id]['config']['name']
        
        # Calculate total area of the zone
        total_area = sum(
            s['area_ha'] for s in class_stats.values() if s['count'] > 0
        )
        
        for class_name, element in class_stats.items():
            if element['count'] > 0:
                all_class_data.append({
                    'Zona': zone_display,
                    'Clase': class_name,
                    'Regiones': element['count'],
                    'Área (ha)': element['area_ha'],
                    '% Área': (element['area_ha'] / total_area) * 100 if total_area > 0 else 0,
                    'NDVI Promedio': element['mean_ndvi'],
                    'NDVI Desv. Est.': element['std_ndvi']
                })
    
    # Create DataFrame
    df_all_classes = pd.DataFrame(all_class_data)
    
    if not return_dataframe:
        return semantic_results, df_all_classes, None
    
    # Format numeric columns for display
    df_all_classes_formatted = df_all_classes.copy()
    df_all_classes_formatted['Área (ha)'] = df_all_classes_formatted['Área (ha)'].apply(
        lambda x: f"{x:.1f}"
    )
    df_all_classes_formatted['% Área'] = df_all_classes_formatted['% Área'].apply(
        lambda x: f"{x:.1f}%"
    )
    df_all_classes_formatted['NDVI Promedio'] = df_all_classes_formatted['NDVI Promedio'].apply(
        lambda x: f"{x:.3f}"
    )
    df_all_classes_formatted['NDVI Desv. Est.'] = df_all_classes_formatted['NDVI Desv. Est.'].apply(
        lambda x: f"±{x:.3f}"
    )
    
    logger.info(f"Classification complete for {len(semantic_results)} zones")
    
    return semantic_results, df_all_classes, df_all_classes_formatted


def print_classification_summary(
    semantic_results: Dict,
    zones_data: Dict
) -> None:
    """
    Print summary of classification results.
    
    Parameters
    ----------
    semantic_results : dict
        Classification results from classify_all_zones()
    zones_data : dict
        Zone configuration data
        
    Examples
    --------
    >>> print_classification_summary(semantic_results, zones_data)
    """
    print("\n" + "="*80)
    print("RESUMEN DE CLASIFICACIÓN SEMÁNTICA")
    print("="*80)
    
    for zone_id, results in semantic_results.items():
        zone_name = zones_data[zone_id]['config']['name']
        class_stats = results['class_stats']
        
        print(f"\n{zone_name}:")
        
        # Count classes with regions
        active_classes = sum(1 for s in class_stats.values() if s['count'] > 0)
        total_regions = sum(s['count'] for s in class_stats.values())
        total_area = sum(s['area_ha'] for s in class_stats.values())
        
        print(f"  Clases detectadas: {active_classes}/6")
        print(f"  Total regiones:    {total_regions}")
        print(f"  Área total:        {total_area:.1f} ha")
        
        # Show top 3 classes by area
        sorted_classes = sorted(
            [(name, stats) for name, stats in class_stats.items() if stats['count'] > 0],
            key=lambda x: x[1]['area_ha'],
            reverse=True
        )[:3]
        
        print(f"  Top 3 clases:")
        for class_name, stats in sorted_classes:
            pct = (stats['area_ha'] / total_area) * 100 if total_area > 0 else 0
            print(f"    - {class_name}: {stats['area_ha']:.1f} ha ({pct:.1f}%)")
    
    print("="*80 + "\n")


def export_classification_results(
    semantic_results: Dict,
    df_all_classes: pd.DataFrame,
    output_dir: Path,
    save_maps: bool = True
) -> None:
    """
    Export classification results to files.

    Parameters
    ----------
    semantic_results : dict
        Classification results from classify_all_zones()
    df_all_classes : pd.DataFrame
        DataFrame with classification data
    output_dir : Path
        Directory to save results
    save_maps : bool, default=True
        Whether to save semantic maps as numpy arrays

    Examples
    --------
    >>> export_classification_results(
    ...     semantic_results,
    ...     df_all_classes,
    ...     Path('output/classification')
    ... )
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save DataFrame
    csv_path = output_dir / 'classification_summary.csv'
    df_all_classes.to_csv(csv_path, index=False)
    logger.info(f"Classification summary saved to: {csv_path}")

    # Save semantic maps
    if save_maps:
        maps_dir = output_dir / 'semantic_maps'
        maps_dir.mkdir(exist_ok=True)

        for zone_id, results in semantic_results.items():
            map_path = maps_dir / f'semantic_map_zone_{zone_id}.npy'
            np.save(map_path, results['semantic_map'])
            logger.info(f"Semantic map saved to: {map_path}")

    logger.info(f"All classification results exported to: {output_dir}")


def analyze_crop_stress(
    semantic_results: Dict,
    zones_data: Dict,
    zone_mapping: Dict,
    crop_class_ids: Optional[list] = None,
    stress_thresholds: Optional[Dict[str, tuple]] = None,
    return_formatted: bool = True
) -> tuple:
    """
    Analyze crop stress levels across all zones based on NDVI values.

    This function filters crop regions and classifies them into stress
    levels based on NDVI thresholds. It calculates area and percentage
    for each stress level per zone and provides aggregated statistics.

    Parameters
    ----------
    semantic_results : dict
        Classification results from classify_all_zones()
        Format: {zone_id: {'classifications': {...}, ...}}
    zones_data : dict
        Zone configuration data
        Format: {zone_id: {'config': {'name': str, ...}, ...}}
    zone_mapping : dict
        Mapping from zone keys to zone IDs
        Format: {zone_name: zone_id}
    crop_class_ids : list, optional
        List of class IDs considered as crops (default: [3, 4])
        3 = Agricultural Crops, 4 = Irrigated Crops
    stress_thresholds : dict, optional
        NDVI thresholds for stress classification
        Default: {
            'Low Stress': (0.5, 0.6),
            'Medium Stress': (0.4, 0.5),
            'High Stress': (0.3, 0.4),
            'Very High Stress': (0.0, 0.3)
        }
    return_formatted : bool, default=True
        Whether to return formatted DataFrames with string formatting

    Returns
    -------
    tuple
        (df_crop_stress_raw, df_crop_stress_formatted, df_total_raw, df_total_formatted)
        - df_crop_stress_raw: pandas DataFrame with raw per-zone data
        - df_crop_stress_formatted: pandas DataFrame with formatted per-zone strings
        - df_total_raw: pandas DataFrame with raw aggregated data
        - df_total_formatted: pandas DataFrame with formatted aggregated strings

    Examples
    --------
    >>> df_stress_raw, df_stress_fmt, df_total_raw, df_total_fmt = analyze_crop_stress(
    ...     semantic_results=semantic_results,
    ...     zones_data=zones_data,
    ...     zone_mapping=zone_mapping
    ... )
    >>> display(df_stress_fmt)
    >>> display(df_total_fmt)
    """
    # Default crop class IDs
    if crop_class_ids is None:
        crop_class_ids = [3, 4]  # Agricultural Crops and Irrigated Crops

    # Default stress thresholds
    if stress_thresholds is None:
        stress_thresholds = {
            'Low Stress': (0.5, 0.6),
            'Medium Stress': (0.4, 0.5),
            'High Stress': (0.3, 0.4),
            'Very High Stress': (0.0, 0.3)
        }

    logger.info("Analyzing crop stress across all zones")

    crop_stress_data = []

    # Analyze each zone
    for zone_name, zone_id in zone_mapping.items():
        if zone_id not in semantic_results:
            logger.warning(f"Zone {zone_name} not found in semantic results, skipping")
            continue

        zone_display = zones_data[zone_id]['config']['name']
        classifications = semantic_results[zone_id]['classifications']

        # Filter crop regions (classifications is a dict: region_id -> ClassificationResult)
        crop_regions = [
            region for region in classifications.values()
            if region.class_id in crop_class_ids
        ]

        if not crop_regions:
            logger.info(f"No crop regions found in {zone_name}")
            continue

        # Initialize stress counters
        stress_stats = {
            stress_level: {'area_ha': 0.0, 'count': 0}
            for stress_level in stress_thresholds.keys()
        }

        # Classify each crop region by stress level
        for region in crop_regions:
            ndvi_mean = region.mean_ndvi
            area_ha = region.area_hectares

            # Determine stress level
            for stress_level, (min_ndvi, max_ndvi) in stress_thresholds.items():
                if min_ndvi <= ndvi_mean < max_ndvi:
                    stress_stats[stress_level]['area_ha'] += area_ha
                    stress_stats[stress_level]['count'] += 1
                    break

        # Calculate total crop area
        total_crop_area = sum(stats['area_ha'] for stats in stress_stats.values())

        # Add to results
        for stress_level, stats in stress_stats.items():
            if stats['area_ha'] > 0:
                crop_stress_data.append({
                    'Zona': zone_display,
                    'Nivel de Estrés': stress_level,
                    'Regiones': stats['count'],
                    'Área (ha)': stats['area_ha'],
                    '% del Total de Cultivos': (stats['area_ha'] / total_crop_area) * 100 if total_crop_area > 0 else 0
                })

    # Create per-zone DataFrame
    df_crop_stress_raw = pd.DataFrame(crop_stress_data)

    # Calculate aggregated totals across all zones
    if len(df_crop_stress_raw) > 0:
        total_data = []
        for stress_level in stress_thresholds.keys():
            stress_rows = df_crop_stress_raw[df_crop_stress_raw['Nivel de Estrés'] == stress_level]
            if len(stress_rows) > 0:
                total_area = stress_rows['Área (ha)'].sum()
                total_regions = stress_rows['Regiones'].sum()
                total_data.append({
                    'Nivel de Estrés': stress_level,
                    'Regiones': total_regions,
                    'Área Total (ha)': total_area
                })

        df_total_raw = pd.DataFrame(total_data)

        # Calculate percentage of total crop area
        total_crop_area_all = df_total_raw['Área Total (ha)'].sum()
        df_total_raw['% del Total'] = (df_total_raw['Área Total (ha)'] / total_crop_area_all) * 100 if total_crop_area_all > 0 else 0
    else:
        df_total_raw = pd.DataFrame()

    if not return_formatted:
        return df_crop_stress_raw, None, df_total_raw, None

    # Format per-zone DataFrame
    df_crop_stress_formatted = df_crop_stress_raw.copy()
    df_crop_stress_formatted['Área (ha)'] = df_crop_stress_formatted['Área (ha)'].apply(
        lambda x: f"{x:.1f}"
    )
    df_crop_stress_formatted['% del Total de Cultivos'] = df_crop_stress_formatted['% del Total de Cultivos'].apply(
        lambda x: f"{x:.1f}%"
    )

    # Format totals DataFrame
    if len(df_total_raw) > 0:
        df_total_formatted = df_total_raw.copy()
        df_total_formatted['Área Total (ha)'] = df_total_formatted['Área Total (ha)'].apply(
            lambda x: f"{x:.1f}"
        )
        df_total_formatted['% del Total'] = df_total_formatted['% del Total'].apply(
            lambda x: f"{x:.1f}%"
        )
    else:
        df_total_formatted = df_total_raw

    logger.info(f"Crop stress analysis complete for {len(zone_mapping)} zones")

    return df_crop_stress_raw, df_crop_stress_formatted, df_total_raw, df_total_formatted
