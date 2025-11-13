"""
Dynamic World downloader utility.

This module provides functions to download Dynamic World land cover masks
for validation purposes. Supports both Google Earth Engine API and manual download.

References:
    - Dynamic World: https://www.dynamicworld.app/
    - Paper: Brown et al. (2022) https://doi.org/10.1038/s41597-022-01307-4
"""

import numpy as np
from pathlib import Path
import logging
from typing import Optional, Tuple, Dict
import warnings

logger = logging.getLogger(__name__)


# Zone definitions matching our Sentinel-2 data
ZONE_CONFIGS = {
    'mexicali': {
        'bbox': [-115.5, 32.3, -115.1, 32.7],  # [west, south, east, north]
        'center': (32.5, -115.3),
        'date': '2024-10-15',
        'name': 'Mexicali'
    },
    'bajio': {
        'bbox': [-101.6, 20.8, -101.2, 21.2],
        'center': (21.0, -101.4),
        'date': '2024-10-15',
        'name': 'Bajío'
    },
    'sinaloa': {
        'bbox': [-108.4, 25.6, -108.0, 26.0],
        'center': (25.8, -108.2),
        'date': '2024-10-15',
        'name': 'Sinaloa'
    }
}


def check_dynamic_world_available(zone_name: str, data_dir: Path) -> bool:
    """
    Check if Dynamic World mask is already downloaded for a zone.
    
    Parameters
    ----------
    zone_name : str
        Name of the zone (mexicali, bajio, sinaloa)
    data_dir : Path
        Directory where Dynamic World masks are stored
        
    Returns
    -------
    bool
        True if mask exists, False otherwise
    """
    # Check both possible filenames
    dw_file1 = data_dir / f'{zone_name}_dw_label.tif'
    dw_file2 = data_dir / f'{zone_name}_dw.tif'
    return dw_file1.exists() or dw_file2.exists()


def get_download_instructions(zone_name: str) -> Dict[str, str]:
    """
    Get manual download instructions for a zone.
    
    Parameters
    ----------
    zone_name : str
        Name of the zone
        
    Returns
    -------
    dict
        Dictionary with download instructions
    """
    if zone_name not in ZONE_CONFIGS:
        raise ValueError(f"Unknown zone: {zone_name}. Available: {list(ZONE_CONFIGS.keys())}")
    
    config = ZONE_CONFIGS[zone_name]
    lat, lon = config['center']
    
    instructions = {
        'zone': zone_name,
        'url': 'https://www.dynamicworld.app/explore/',
        'coordinates': f"{lat}°N, {abs(lon)}°W",
        'date': config['date'],
        'steps': [
            f"1. Go to: https://www.dynamicworld.app/explore/",
            f"2. Navigate to coordinates: {lat}°N, {abs(lon)}°W",
            f"3. Select date: {config['date']} (±5 days is OK)",
            f"4. Click 'Download' → 'Label' (most likely class)",
            f"5. Format: GeoTIFF",
            f"6. Save as: {zone_name}_dw_label.tif"
        ]
    }
    
    return instructions


def print_download_instructions(zone_name: Optional[str] = None):
    """
    Print download instructions for one or all zones.
    
    Parameters
    ----------
    zone_name : str, optional
        Specific zone name, or None for all zones
    """
    zones = [zone_name] if zone_name else list(ZONE_CONFIGS.keys())
    
    print("\n" + "="*70)
    print("DYNAMIC WORLD MANUAL DOWNLOAD INSTRUCTIONS")
    print("="*70)
    
    for zone in zones:
        instructions = get_download_instructions(zone)
        print(f"\n{instructions['zone'].upper()}:")
        for step in instructions['steps']:
            print(f"  {step}")
    
    print("\n" + "="*70)
    print("After downloading, place files in: data/dynamic_world/")
    print("="*70)


def download_with_gee(zone_name: str, output_dir: Path) -> bool:
    """
    Download Dynamic World mask using Google Earth Engine API.
    
    Requires:
        - earthengine-api installed
        - GEE account authenticated
    
    Parameters
    ----------
    zone_name : str
        Name of the zone
    output_dir : Path
        Output directory
        
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    try:
        import ee
    except ImportError:
        logger.error("earthengine-api not installed. Install with: poetry add earthengine-api")
        return False
    
    try:
        ee.Initialize()
    except Exception as e:
        logger.error(f"Failed to initialize GEE: {e}")
        logger.info("Run: earthengine authenticate")
        return False
    
    if zone_name not in ZONE_CONFIGS:
        logger.error(f"Unknown zone: {zone_name}")
        return False
    
    config = ZONE_CONFIGS[zone_name]
    logger.info(f"Downloading Dynamic World for {zone_name} via GEE...")
    
    try:
        # Create region of interest
        roi = ee.Geometry.Rectangle(config['bbox'])
        
        # Load Dynamic World collection
        dw = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
        
        # Filter by date
        start_date = ee.Date(config['date']).advance(-5, 'day')
        end_date = ee.Date(config['date']).advance(5, 'day')
        
        dw_filtered = dw.filterDate(start_date, end_date).filterBounds(roi)
        
        # Get most recent image
        dw_image = dw_filtered.sort('system:time_start', False).first()
        
        # Get label band
        label = dw_image.select('label')
        
        # Generate download URL
        url = label.getDownloadURL({
            'region': roi,
            'scale': 10,
            'format': 'GEO_TIFF'
        })
        
        logger.info(f"Download URL generated: {url[:100]}...")
        logger.info(f"Please download manually and save to: {output_dir / f'{zone_name}_dw_label.tif'}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to download {zone_name}: {e}")
        return False


def generate_synthetic_gt_from_ndvi(
    ndvi: np.ndarray,
    seed: int = 42
) -> np.ndarray:
    """
    Generate synthetic ground truth from NDVI for testing.
    
    This is a fallback when Dynamic World is not available.
    Uses simple NDVI thresholds to classify land cover.
    
    Parameters
    ----------
    ndvi : np.ndarray
        NDVI image (H, W)
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    np.ndarray
        Synthetic ground truth mask (H, W) with classes 0-4
    """
    np.random.seed(seed)
    h, w = ndvi.shape
    gt = np.zeros((h, w), dtype=np.int32)
    
    # Water: NDVI < 0
    gt[ndvi < 0] = 0
    
    # Crops: NDVI > 0.4 (healthy vegetation)
    gt[ndvi > 0.4] = 1
    
    # Bare Soil: 0 < NDVI < 0.2
    gt[(ndvi >= 0) & (ndvi < 0.2)] = 3
    
    # Other: 0.2 <= NDVI <= 0.4
    gt[(ndvi >= 0.2) & (ndvi <= 0.4)] = 4
    
    # Urban: random patches with low NDVI
    urban_mask = np.random.rand(h, w) < 0.05
    urban_mask &= (ndvi < 0.3)
    gt[urban_mask] = 2
    
    logger.info(f"Generated synthetic GT: {np.unique(gt, return_counts=True)}")
    
    return gt


def load_or_generate_ground_truth(
    zone_name: str,
    ndvi: np.ndarray,
    data_dir: Path,
    use_synthetic: bool = False,
    prefer_esa: bool = True
) -> Tuple[np.ndarray, bool]:
    """
    Load ground truth from Dynamic World, ESA WorldCover, or generate synthetic.
    
    Priority order (if prefer_esa=True):
    1. ESA WorldCover (if available)
    2. Dynamic World (if available)
    3. Synthetic (fallback)
    
    Parameters
    ----------
    zone_name : str
        Name of the zone
    ndvi : np.ndarray
        NDVI image for synthetic generation
    data_dir : Path
        Directory with ground truth masks
    use_synthetic : bool
        Force use of synthetic GT
    prefer_esa : bool
        Prefer ESA WorldCover over Dynamic World
        
    Returns
    -------
    gt_mask : np.ndarray
        Ground truth mask (H, W)
    is_synthetic : bool
        True if synthetic, False if real data
    """
    if use_synthetic:
        logger.info("Using synthetic ground truth (forced)")
        gt_mask = generate_synthetic_gt_from_ndvi(ndvi)
        return gt_mask, True
    
    # Try ESA WorldCover first (if preferred)
    if prefer_esa:
        esa_dir = data_dir.parent / 'esa_worldcover'
        esa_file = esa_dir / f'{zone_name}_esa_worldcover.tif'
        
        if esa_file.exists():
            logger.info(f"Loading ESA WorldCover: {esa_file}")
            try:
                from .esa_worldcover_loader import load_esa_worldcover
                gt_mask, success = load_esa_worldcover(zone_name, ndvi.shape, esa_dir)
                if success:
                    logger.info(f"Loaded ESA WorldCover GT: {gt_mask.shape}")
                    return gt_mask, False
            except Exception as e:
                logger.error(f"Failed to load ESA WorldCover: {e}")
    
    # Try Dynamic World (check both possible filenames)
    dw_file1 = data_dir / f'{zone_name}_dw_label.tif'
    dw_file2 = data_dir / f'{zone_name}_dw.tif'
    dw_file = dw_file1 if dw_file1.exists() else dw_file2 if dw_file2.exists() else None
    
    if dw_file and dw_file.exists():
        logger.info(f"Loading Dynamic World mask: {dw_file}")
        
        try:
            import rasterio
            from scipy.ndimage import zoom
            
            with rasterio.open(dw_file) as src:
                dw_mask = src.read(1)
                
                # Resize to match NDVI if needed
                if dw_mask.shape != ndvi.shape:
                    logger.info(f"Resizing DW mask from {dw_mask.shape} to {ndvi.shape}")
                    zoom_factors = (ndvi.shape[0] / dw_mask.shape[0],
                                   ndvi.shape[1] / dw_mask.shape[1])
                    dw_mask = zoom(dw_mask, zoom_factors, order=0)
            
            # Map to our classes
            gt_mask = map_dw_to_our_classes(dw_mask)
            logger.info(f"Loaded real Dynamic World GT: {gt_mask.shape}")
            return gt_mask, False
            
        except Exception as e:
            logger.error(f"Failed to load Dynamic World: {e}")
    
    # Fallback to synthetic
    logger.warning(f"No real ground truth found for {zone_name}")
    logger.warning("Using synthetic ground truth")
    
    gt_mask = generate_synthetic_gt_from_ndvi(ndvi)
    return gt_mask, True


def map_dw_to_our_classes(dw_mask: np.ndarray) -> np.ndarray:
    """
    Map Dynamic World classes to our taxonomy.
    
    Dynamic World classes:
        0: Water
        1: Trees
        2: Grass
        3: Flooded Vegetation
        4: Crops
        5: Shrub & Scrub
        6: Built Area
        7: Bare Ground
        8: Snow & Ice
    
    Our classes:
        0: Water
        1: Crop
        2: Urban
        3: Bare Soil
        4: Other
    
    Parameters
    ----------
    dw_mask : np.ndarray
        Dynamic World mask with classes 0-8
        
    Returns
    -------
    np.ndarray
        Mapped mask with our classes 0-4
    """
    mapping = {
        0: 0,  # Water → Water
        1: 4,  # Trees → Other
        2: 4,  # Grass → Other
        3: 4,  # Flooded Vegetation → Other
        4: 1,  # Crops → Crop
        5: 4,  # Shrub & Scrub → Other
        6: 2,  # Built Area → Urban
        7: 3,  # Bare Ground → Bare Soil
        8: 4   # Snow & Ice → Other
    }
    
    our_mask = np.zeros_like(dw_mask)
    for dw_class, our_class in mapping.items():
        our_mask[dw_mask == dw_class] = our_class
    
    return our_mask


def check_all_zones(data_dir: Path) -> Dict[str, bool]:
    """
    Check which zones have Dynamic World data available.
    
    Parameters
    ----------
    data_dir : Path
        Directory with Dynamic World masks
        
    Returns
    -------
    dict
        Dictionary mapping zone_name -> availability (bool)
    """
    availability = {}
    for zone_name in ZONE_CONFIGS.keys():
        availability[zone_name] = check_dynamic_world_available(zone_name, data_dir)
    
    return availability


if __name__ == '__main__':
    # Print download instructions
    print_download_instructions()
