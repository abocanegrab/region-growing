"""
Download HLS images from Sentinel-2 for specific zones in Mexico.

This script downloads HLS-format images (6 bands) from three agricultural
zones in Mexico: Mexicali, Bajío, and Sinaloa.

Usage:
    python scripts/download_hls_image.py --zone mexicali
    python scripts/download_hls_image.py --zone bajio --date-from 2024-01-01
    python scripts/download_hls_image.py --all
"""
import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta
import os
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from src.utils.sentinel_download import create_sentinel_config, download_hls_bands
from src.features.hls_processor import prepare_hls_image, save_embeddings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


MEXICO_TEST_ZONES = {
    'mexicali': {
        'name': 'Valle de Mexicali, Baja California',
        'bbox': {
            'min_lat': 32.45,
            'min_lon': -115.35,
            'max_lat': 32.55,
            'max_lon': -115.25
        },
        'description': 'Agricultura intensiva de riego - Trigo, algodón',
        'expected_stress': 'variable (depende de riego)'
    },
    'bajio': {
        'name': 'Bajío, Guanajuato',
        'bbox': {
            'min_lat': 20.85,
            'min_lon': -101.45,
            'max_lat': 20.95,
            'max_lon': -101.35
        },
        'description': 'Agricultura diversa - Maíz, sorgo, hortalizas',
        'expected_stress': 'bajo a medio'
    },
    'sinaloa': {
        'name': 'Valle de Culiacán, Sinaloa',
        'bbox': {
            'min_lat': 24.75,
            'min_lon': -107.45,
            'max_lat': 24.85,
            'max_lon': -107.35
        },
        'description': 'Agricultura de exportación - Tomate, chile',
        'expected_stress': 'bajo (riego tecnificado)'
    }
}


def load_credentials():
    """
    Load Sentinel Hub credentials from environment or config file.
    """
    client_id = os.getenv('SENTINELHUB_CLIENT_ID')
    client_secret = os.getenv('SENTINELHUB_CLIENT_SECRET')
    
    if not client_id or not client_secret:
        credentials_file = Path(__file__).parent.parent / 'sentinelhub-secrets_.txt'
        if credentials_file.exists():
            logger.info(f"Loading credentials from {credentials_file}")
            with open(credentials_file, 'r') as f:
                lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                if len(lines) >= 2:
                    client_id = lines[0]
                    client_secret = lines[1]
    
    if not client_id or not client_secret:
        raise ValueError(
            "Sentinel Hub credentials not found. "
            "Set SENTINELHUB_CLIENT_ID and SENTINELHUB_CLIENT_SECRET environment variables "
            "or add them to sentinelhub-secrets_.txt"
        )
    
    return client_id, client_secret


def download_zone(
    zone_name: str,
    date_from: str = None,
    date_to: str = None,
    output_dir: Path = None
):
    """
    Download HLS image for a specific zone.
    
    Parameters
    ----------
    zone_name : str
        Zone name ('mexicali', 'bajio', or 'sinaloa')
    date_from : str, optional
        Start date in YYYY-MM-DD format
    date_to : str, optional
        End date in YYYY-MM-DD format
    output_dir : Path, optional
        Output directory (default: img/sentinel2/mexico/{zone_name})
    """
    if zone_name not in MEXICO_TEST_ZONES:
        raise ValueError(f"Unknown zone: {zone_name}. Must be one of {list(MEXICO_TEST_ZONES.keys())}")
    
    zone_info = MEXICO_TEST_ZONES[zone_name]
    
    logger.info("="*70)
    logger.info(f"Downloading HLS image for: {zone_info['name']}")
    logger.info(f"Description: {zone_info['description']}")
    logger.info(f"Expected stress: {zone_info['expected_stress']}")
    logger.info("="*70)
    
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / 'img' / 'sentinel2' / 'mexico' / zone_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if date_to is None:
        date_to = datetime.now().strftime('%Y-%m-%d')
    if date_from is None:
        date_from_dt = datetime.strptime(date_to, '%Y-%m-%d') - timedelta(days=30)
        date_from = date_from_dt.strftime('%Y-%m-%d')
    
    logger.info(f"Date range: {date_from} to {date_to}")
    logger.info(f"BBox: {zone_info['bbox']}")
    
    client_id, client_secret = load_credentials()
    config = create_sentinel_config(client_id, client_secret)
    
    logger.info("Downloading HLS bands from Sentinel Hub...")
    try:
        data = download_hls_bands(
            bbox_coords=zone_info['bbox'],
            config=config,
            date_from=date_from,
            date_to=date_to,
            resolution=10,
            max_cloud_coverage=0.3
        )
            
    except ValueError as e:
        # Data validation error from download_hls_bands
        logger.error(f"Data validation failed: {e}")
        logger.error(f"\nPossible solutions:")
        logger.error(f"  1. Try a different date range (e.g., last 60 days)")
        logger.error(f"  2. Increase max_cloud_coverage to 0.5 or 0.7")
        logger.error(f"  3. Verify the area at https://apps.sentinel-hub.com/eo-browser/")
        raise
    except Exception as e:
        # Other errors (connection, auth, etc.)
        logger.error(f"Failed to download data: {e}")
        logger.error("Check credentials and internet connection")
        raise
    
    logger.info(f"Downloaded successfully!")
    logger.info(f"  10m bands: {list(data['bands_10m'].keys())}")
    logger.info(f"  20m bands: {list(data['bands_20m'].keys())}")
    logger.info(f"  Dimensions 10m: {data['metadata']['dimensions_10m']}")
    logger.info(f"  Dimensions 20m: {data['metadata']['dimensions_20m']}")
    
    logger.info("Saving bands to disk...")
    for band_name, band_data in data['bands_10m'].items():
        output_path = output_dir / f"{band_name}_10m.npy"
        np.save(output_path, band_data)
        logger.info(f"  Saved {band_name}: {output_path}")
    
    for band_name, band_data in data['bands_20m'].items():
        output_path = output_dir / f"{band_name}_20m.npy"
        np.save(output_path, band_data)
        logger.info(f"  Saved {band_name}: {output_path}")
    
    logger.info("Preparing HLS image...")
    hls_image = prepare_hls_image(data['bands_10m'], data['bands_20m'])
    logger.info(f"  HLS image shape: {hls_image.shape}")
    
    hls_output_path = output_dir / "hls_image.npy"
    np.save(hls_output_path, hls_image)
    logger.info(f"  Saved HLS image: {hls_output_path}")
    
    metadata_path = output_dir / "metadata.txt"
    with open(metadata_path, 'w') as f:
        f.write(f"Zone: {zone_info['name']}\n")
        f.write(f"Description: {zone_info['description']}\n")
        f.write(f"BBox: {zone_info['bbox']}\n")
        f.write(f"Date range: {date_from} to {date_to}\n")
        f.write(f"Dimensions 10m: {data['metadata']['dimensions_10m']}\n")
        f.write(f"Dimensions 20m: {data['metadata']['dimensions_20m']}\n")
        f.write(f"Downloaded: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    logger.info(f"  Saved metadata: {metadata_path}")
    
    logger.info("="*70)
    logger.info(f"Download complete for {zone_name}!")
    logger.info(f"Files saved to: {output_dir}")
    logger.info("="*70)
    
    return {
        'zone': zone_name,
        'zone_info': zone_info,
        'output_dir': output_dir,
        'hls_image_shape': hls_image.shape,
        'metadata': data['metadata']
    }


def main():
    parser = argparse.ArgumentParser(
        description='Download HLS images from Sentinel-2 for zones in Mexico'
    )
    parser.add_argument(
        '--zone',
        type=str,
        choices=['mexicali', 'bajio', 'sinaloa'],
        help='Zone to download'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Download all zones'
    )
    parser.add_argument(
        '--date-from',
        type=str,
        help='Start date (YYYY-MM-DD). Default: 30 days ago'
    )
    parser.add_argument(
        '--date-to',
        type=str,
        help='End date (YYYY-MM-DD). Default: today'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        help='Output directory. Default: img/sentinel2/mexico/{zone}'
    )
    
    args = parser.parse_args()
    
    if not args.zone and not args.all:
        parser.error('Either --zone or --all must be specified')
    
    if args.all:
        zones_to_download = list(MEXICO_TEST_ZONES.keys())
    else:
        zones_to_download = [args.zone]
    
    logger.info(f"Starting download for zones: {zones_to_download}")
    
    results = []
    for zone in zones_to_download:
        try:
            result = download_zone(
                zone_name=zone,
                date_from=args.date_from,
                date_to=args.date_to,
                output_dir=args.output_dir
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to download {zone}: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info("\n" + "="*70)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("="*70)
    for result in results:
        logger.info(f"  {result['zone']}: SUCCESS ({result['hls_image_shape']})")
    logger.info(f"\nTotal zones downloaded: {len(results)}/{len(zones_to_download)}")
    logger.info("="*70)


if __name__ == '__main__':
    main()
