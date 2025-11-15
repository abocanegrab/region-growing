"""
Test zones configuration for Mexico agricultural regions.

This module contains predefined test zones for agricultural analysis in Mexico,
including bounding boxes, descriptions, and typical crops for each region.
"""

# Mexico test zones with agricultural characteristics
MEXICO_TEST_ZONES = {
    'mexicali': {
        'name': 'Valle de Mexicali',
        'bbox': [32.4, -115.4, 32.5, -115.3],
        'description': 'Valle de Mexicali - Agricultura intensiva de riego',
        'crops': 'Trigo, algodón, alfalfa',
        'state': 'Baja California',
        'climate': 'Árido',
        'irrigation': 'Intensivo'
    },
    'bajio': {
        'name': 'El Bajío',
        'bbox': [20.8, -101.5, 20.9, -101.4],
        'description': 'El Bajío - Región agrícola diversificada',
        'crops': 'Sorgo, maíz, hortalizas',
        'state': 'Guanajuato',
        'climate': 'Templado',
        'irrigation': 'Mixto'
    },
    'sinaloa': {
        'name': 'Valle de Culiacán',
        'bbox': [24.7, -107.5, 24.8, -107.4],
        'description': 'Valle de Culiacán - Agricultura de exportación',
        'crops': 'Tomate, chile, maíz',
        'state': 'Sinaloa',
        'climate': 'Tropical',
        'irrigation': 'Tecnificado'
    }
}


def get_zone_info(zone_key: str) -> dict:
    """
    Get information for a specific test zone.
    
    Parameters
    ----------
    zone_key : str
        Zone identifier ('mexicali', 'bajio', or 'sinaloa')
        
    Returns
    -------
    dict
        Zone configuration dictionary
        
    Raises
    ------
    KeyError
        If zone_key is not found in MEXICO_TEST_ZONES
    """
    if zone_key not in MEXICO_TEST_ZONES:
        raise KeyError(f"Zone '{zone_key}' not found. Available zones: {list(MEXICO_TEST_ZONES.keys())}")
    
    return MEXICO_TEST_ZONES[zone_key]


def get_all_zones() -> dict:
    """
    Get all available test zones.
    
    Returns
    -------
    dict
        Dictionary with all test zones
    """
    return MEXICO_TEST_ZONES


def get_zone_names() -> list:
    """
    Get list of all zone names.
    
    Returns
    -------
    list
        List of zone display names
    """
    return [zone['name'] for zone in MEXICO_TEST_ZONES.values()]


def get_zone_keys() -> list:
    """
    Get list of all zone keys.
    
    Returns
    -------
    list
        List of zone keys ('mexicali', 'bajio', 'sinaloa')
    """
    return list(MEXICO_TEST_ZONES.keys())


def load_or_download_zones(
    date_from: str,
    date_to: str,
    project_root,
    sentinel_client_id: str = None,
    sentinel_client_secret: str = None,
    zones_to_process: list = None
) -> dict:
    """
    Load or download HLS images for all test zones.
    
    This function checks if HLS images already exist locally. If they do,
    it loads them. If not, it downloads them from Sentinel Hub.
    
    Parameters
    ----------
    date_from : str
        Start date in format 'YYYY-MM-DD'
    date_to : str
        End date in format 'YYYY-MM-DD'
    project_root : Path
        Root directory of the project
    sentinel_client_id : str, optional
        Sentinel Hub client ID (can be None if loading existing data)
    sentinel_client_secret : str, optional
        Sentinel Hub client secret (can be None if loading existing data)
    zones_to_process : list, optional
        List of zone keys to process. If None, processes all zones.
        
    Returns
    -------
    dict
        Dictionary with zone data:
        {
            'zone_id': {
                'config': zone_config,
                'hls_image': numpy array,
                'metadata': {'zone_id': str, 'name': str}
            }
        }
        
    Examples
    --------
    >>> from pathlib import Path
    >>> zones_data = load_or_download_zones(
    ...     date_from='2024-01-15',
    ...     date_to='2024-01-15',
    ...     project_root=Path('../../'),
    ...     sentinel_client_id='your_id',
    ...     sentinel_client_secret='your_secret'
    ... )
    >>> print(f"Loaded {len(zones_data)} zones")
    
    >>> # Load only specific zones
    >>> zones_data = load_or_download_zones(
    ...     date_from='2024-01-15',
    ...     date_to='2024-01-15',
    ...     project_root=Path('../../'),
    ...     zones_to_process=['mexicali', 'bajio']
    ... )
    """
    import numpy as np
    import os
    from pathlib import Path
    
    # Import required functions
    try:
        from src.utils.sentinel_download import (
            create_sentinel_config,
            download_hls_bands
        )
        from src.features.hls_processor import prepare_hls_image
    except ImportError as e:
        raise ImportError(
            f"Required functions not found: {str(e)}. "
            "Make sure sentinel_download and hls_processor modules exist."
        )
    
    # Determine which zones to process
    zones_to_load = zones_to_process if zones_to_process else list(MEXICO_TEST_ZONES.keys())
    
    zones_data = {}
    
    for zone_id in zones_to_load:
        if zone_id not in MEXICO_TEST_ZONES:
            print(f"Warning: Zone '{zone_id}' not found in MEXICO_TEST_ZONES. Skipping.")
            continue
        
        zone_config = MEXICO_TEST_ZONES[zone_id]
        print(f"\nProcessing {zone_config['name']}...")
        
        zone_dir = project_root / 'img' / 'sentinel2' / 'mexico' / zone_id
        hls_image_path = zone_dir / 'hls_image.npy'
        
        try:
            # Check if HLS image already exists
            if hls_image_path.exists():
                print(f"  Loading existing HLS image from {hls_image_path}")
                hls_image = np.load(hls_image_path)
                print(f"  Loaded {zone_config['name']}: {hls_image.shape}")
            else:
                # Download HLS bands if not exists
                if not sentinel_client_id or not sentinel_client_secret:
                    print(f"  Error: HLS image not found and no Sentinel credentials provided.")
                    print(f"  Skipping {zone_config['name']}")
                    zones_data[zone_id] = None
                    continue
                
                print(f"  Downloading HLS bands for {zone_config['name']}...")
                config = create_sentinel_config(
                    client_id=sentinel_client_id,
                    client_secret=sentinel_client_secret
                )
                
                # Convert bbox list to dict format expected by download_hls_bands
                bbox_dict = {
                    'min_lat': zone_config['bbox'][0],
                    'min_lon': zone_config['bbox'][1],
                    'max_lat': zone_config['bbox'][2],
                    'max_lon': zone_config['bbox'][3]
                }
                
                result = download_hls_bands(
                    bbox_coords=bbox_dict,
                    config=config,
                    date_from=date_from,
                    date_to=date_to
                )
                
                # Prepare HLS image (resample 20m -> 10m and stack)
                hls_image = prepare_hls_image(
                    bands_10m=result['bands_10m'],
                    bands_20m=result['bands_20m']
                )
                
                # Save for future use
                zone_dir.mkdir(parents=True, exist_ok=True)
                np.save(hls_image_path, hls_image)
                print(f"  Downloaded and saved {zone_config['name']}: {hls_image.shape}")
            
            zones_data[zone_id] = {
                'config': zone_config,
                'hls_image': hls_image,
                'metadata': {
                    'zone_id': zone_id,
                    'name': zone_config['name'],
                    'date_from': date_from,
                    'date_to': date_to
                }
            }
            
        except Exception as e:
            print(f"  Error processing {zone_config['name']}: {e}")
            zones_data[zone_id] = None
    
    successful_zones = len([z for z in zones_data.values() if z is not None])
    print(f"\nProcessing complete: {successful_zones}/{len(zones_to_load)} zones loaded")
    
    return zones_data
