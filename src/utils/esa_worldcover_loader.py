"""
ESA WorldCover loader utility.

This module provides functions to load and process ESA WorldCover data
as an alternative to Dynamic World for ground truth validation.

ESA WorldCover provides 11 land cover classes at 10m resolution for 2021.

References:
    - ESA WorldCover: https://esa-worldcover.org/
    - Viewer: https://viewer.esa-worldcover.org/
"""

import numpy as np
import rasterio
from pathlib import Path
import logging
from typing import Tuple
from scipy.ndimage import zoom

logger = logging.getLogger(__name__)


# ESA WorldCover classes (11 classes)
ESA_CLASSES = {
    10: 'Tree cover',
    20: 'Shrubland',
    30: 'Grassland',
    40: 'Cropland',
    50: 'Built-up',
    60: 'Bare / sparse vegetation',
    70: 'Snow and ice',
    80: 'Permanent water bodies',
    90: 'Herbaceous wetland',
    95: 'Mangroves',
    100: 'Moss and lichen'
}

# Mapping ESA WorldCover → Our classes
ESA_TO_OUR_MAPPING = {
    10: 4,   # Tree cover → Other
    20: 4,   # Shrubland → Other
    30: 4,   # Grassland → Other
    40: 1,   # Cropland → Crop
    50: 2,   # Built-up → Urban
    60: 3,   # Bare/sparse → Bare Soil
    70: 4,   # Snow/ice → Other
    80: 0,   # Water → Water
    90: 4,   # Wetland → Other
    95: 4,   # Mangroves → Other
    100: 4   # Moss/lichen → Other
}


def map_esa_to_our_classes(esa_mask: np.ndarray) -> np.ndarray:
    """
    Map ESA WorldCover classes to our taxonomy.
    
    ESA WorldCover classes:
        10: Tree cover
        20: Shrubland
        30: Grassland
        40: Cropland
        50: Built-up
        60: Bare / sparse vegetation
        70: Snow and ice
        80: Permanent water bodies
        90: Herbaceous wetland
        95: Mangroves
        100: Moss and lichen
    
    Our classes:
        0: Water
        1: Crop
        2: Urban
        3: Bare Soil
        4: Other
    
    Parameters
    ----------
    esa_mask : np.ndarray
        ESA WorldCover mask with classes 10-100
        
    Returns
    -------
    np.ndarray
        Mapped mask with our classes 0-4
    """
    our_mask = np.zeros_like(esa_mask, dtype=np.int32)
    
    for esa_class, our_class in ESA_TO_OUR_MAPPING.items():
        our_mask[esa_mask == esa_class] = our_class
    
    return our_mask


def load_esa_worldcover(
    zone_name: str,
    target_shape: Tuple[int, int],
    data_dir: Path
) -> Tuple[np.ndarray, bool]:
    """
    Load ESA WorldCover mask for a zone.
    
    Parameters
    ----------
    zone_name : str
        Name of the zone (mexicali, bajio, sinaloa)
    target_shape : tuple
        Target shape (H, W) to resize to
    data_dir : Path
        Directory containing ESA WorldCover files
        
    Returns
    -------
    gt_mask : np.ndarray
        Ground truth mask (H, W) with our classes 0-4
    success : bool
        True if loaded from ESA, False if not found
    """
    esa_file = data_dir / f'{zone_name}_esa_worldcover.tif'
    
    if not esa_file.exists():
        logger.warning(f"ESA WorldCover file not found: {esa_file}")
        return None, False
    
    logger.info(f"Loading ESA WorldCover: {esa_file}")
    
    try:
        with rasterio.open(esa_file) as src:
            esa_mask = src.read(1)
            
            logger.info(f"  Original shape: {esa_mask.shape}")
            logger.info(f"  Classes found: {np.unique(esa_mask)}")
            
            # Resize to match target shape if needed
            if esa_mask.shape != target_shape:
                logger.info(f"  Resizing to: {target_shape}")
                zoom_factors = (
                    target_shape[0] / esa_mask.shape[0],
                    target_shape[1] / esa_mask.shape[1]
                )
                esa_mask = zoom(esa_mask, zoom_factors, order=0)  # Nearest neighbor
            
            # Map to our classes
            our_mask = map_esa_to_our_classes(esa_mask)
            
            logger.info(f"  Mapped classes: {np.unique(our_mask)}")
            logger.info(f"  Final shape: {our_mask.shape}")
            
            return our_mask, True
            
    except Exception as e:
        logger.error(f"Failed to load ESA WorldCover: {e}")
        return None, False


def check_esa_available(zone_name: str, data_dir: Path) -> bool:
    """
    Check if ESA WorldCover is available for a zone.
    
    Parameters
    ----------
    zone_name : str
        Name of the zone
    data_dir : Path
        Directory with ESA WorldCover files
        
    Returns
    -------
    bool
        True if available, False otherwise
    """
    esa_file = data_dir / f'{zone_name}_esa_worldcover.tif'
    return esa_file.exists()


def print_esa_download_instructions():
    """Print instructions for downloading ESA WorldCover."""
    print("\n" + "="*70)
    print("ESA WORLDCOVER DOWNLOAD INSTRUCTIONS")
    print("="*70)
    
    zones_info = {
        'mexicali': {
            'coords': '32.5°N, 115.3°W',
            'tiles': 'N42W117 or N36W117',
            'location': 'Baja California (frontera con EE.UU.)'
        },
        'bajio': {
            'coords': '21.0°N, 101.4°W',
            'tiles': 'N21W102 or N18W102',
            'location': 'Guanajuato/Querétaro (centro de México)'
        },
        'sinaloa': {
            'coords': '25.8°N, 108.2°W',
            'tiles': 'N24W108 or N27W108',
            'location': 'Culiacán/Mazatlán (costa oeste)'
        }
    }
    
    print("\n1. Go to: https://viewer.esa-worldcover.org/")
    print("2. Navigate to each zone and download the tile:")
    
    for zone, info in zones_info.items():
        print(f"\n   {zone.upper()}:")
        print(f"     Location: {info['location']}")
        print(f"     Coordinates: {info['coords']}")
        print(f"     Tile ID: {info['tiles']}")
        print(f"     Save as: {zone}_esa_worldcover.tif")
    
    print("\n3. Download settings:")
    print("   - Format: GeoTIFF")
    print("   - Resolution: 10m")
    print("   - Year: 2021")
    
    print("\n4. Save files to: data/esa_worldcover/")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    print_esa_download_instructions()
