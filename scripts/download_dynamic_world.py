"""
Download Dynamic World masks for validation zones.

This script downloads Dynamic World land cover classifications for the three
validation zones (Mexicali, Bajío, Sinaloa) using Google Earth Engine.

Requirements:
    - earthengine-api installed: poetry add earthengine-api
    - GEE account authenticated: earthengine authenticate

Usage:
    poetry run python scripts/download_dynamic_world.py
"""

import ee
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from pathlib import Path
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Zone definitions (matching Sentinel-2 data)
ZONES = {
    'mexicali': {
        'bbox': [-115.5, 32.3, -115.1, 32.7],  # [west, south, east, north]
        'date': '2024-10-15',
        'name': 'Mexicali'
    },
    'bajio': {
        'bbox': [-101.6, 20.8, -101.2, 21.2],
        'date': '2024-10-15',
        'name': 'Bajío'
    },
    'sinaloa': {
        'bbox': [-108.4, 25.6, -108.0, 26.0],
        'date': '2024-10-15',
        'name': 'Sinaloa'
    }
}


def initialize_gee():
    """Initialize Google Earth Engine."""
    try:
        ee.Initialize()
        logger.info("Google Earth Engine initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize GEE: {e}")
        logger.info("Run: earthengine authenticate")
        raise


def download_dynamic_world_mask(zone_name: str, bbox: list, date: str, output_dir: Path):
    """
    Download Dynamic World mask for a specific zone.
    
    Parameters
    ----------
    zone_name : str
        Name of the zone (e.g., 'mexicali')
    bbox : list
        Bounding box [west, south, east, north] in WGS84
    date : str
        Date in format 'YYYY-MM-DD'
    output_dir : Path
        Output directory for GeoTIFF
    """
    logger.info(f"Downloading Dynamic World for {zone_name}...")
    
    # Create region of interest
    roi = ee.Geometry.Rectangle(bbox)
    
    # Load Dynamic World collection
    dw = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
    
    # Filter by date (±5 days window)
    start_date = ee.Date(date).advance(-5, 'day')
    end_date = ee.Date(date).advance(5, 'day')
    
    dw_filtered = dw.filterDate(start_date, end_date).filterBounds(roi)
    
    # Get the most recent image
    dw_image = dw_filtered.sort('system:time_start', False).first()
    
    # Get the label band (most likely class)
    label = dw_image.select('label')
    
    # Get image info
    info = dw_image.getInfo()
    actual_date = datetime.fromtimestamp(info['properties']['system:time_start'] / 1000)
    logger.info(f"  Using image from: {actual_date.strftime('%Y-%m-%d')}")
    
    # Download as numpy array
    try:
        # Get the data
        url = label.getDownloadURL({
            'region': roi,
            'scale': 10,  # 10m resolution
            'format': 'GEO_TIFF'
        })
        
        logger.info(f"  Download URL generated: {url[:100]}...")
        logger.info(f"  Please download manually from the URL above")
        logger.info(f"  Save to: {output_dir / f'{zone_name}_dw_label.tif'}")
        
        return url
        
    except Exception as e:
        logger.error(f"Failed to download {zone_name}: {e}")
        raise


def download_all_zones():
    """Download Dynamic World masks for all validation zones."""
    # Create output directory
    output_dir = Path('data/dynamic_world')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize GEE
    initialize_gee()
    
    # Download each zone
    urls = {}
    for zone_name, zone_info in ZONES.items():
        try:
            url = download_dynamic_world_mask(
                zone_name,
                zone_info['bbox'],
                zone_info['date'],
                output_dir
            )
            urls[zone_name] = url
        except Exception as e:
            logger.error(f"Failed to process {zone_name}: {e}")
            continue
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("DOWNLOAD URLS GENERATED")
    logger.info("="*60)
    for zone_name, url in urls.items():
        logger.info(f"\n{zone_name.upper()}:")
        logger.info(f"  {url}")
    
    logger.info("\n" + "="*60)
    logger.info("MANUAL DOWNLOAD INSTRUCTIONS")
    logger.info("="*60)
    logger.info("1. Copy each URL to your browser")
    logger.info("2. Download the GeoTIFF file")
    logger.info(f"3. Save to: {output_dir.absolute()}/")
    logger.info("4. Rename files to: <zone>_dw_label.tif")


if __name__ == '__main__':
    download_all_zones()
