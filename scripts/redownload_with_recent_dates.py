"""
Re-download Sentinel-2 images with recent dates that have data available.
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta
sys.path.append(str(Path(__file__).parent.parent))

from scripts.download_hls_image import download_zone, MEXICO_TEST_ZONES

def main():
    print("Re-downloading Sentinel-2 images with recent dates...")
    print("=" * 70)
    
    # Use last 30 days to ensure we get data
    date_to = datetime.now()
    date_from = date_to - timedelta(days=30)
    
    date_from_str = date_from.strftime('%Y-%m-%d')
    date_to_str = date_to.strftime('%Y-%m-%d')
    
    print(f"Date range: {date_from_str} to {date_to_str}")
    print("This will download the most recent available image in this range.")
    print()
    
    zones = list(MEXICO_TEST_ZONES.keys())
    results = []
    
    for zone in zones:
        print(f"\nDownloading {zone}...")
        try:
            result = download_zone(
                zone_name=zone,
                date_from=date_from_str,
                date_to=date_to_str
            )
            results.append(result)
            print(f"SUCCESS: {zone}")
        except Exception as e:
            print(f"FAILED: {zone} - {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Successfully downloaded: {len(results)}/{len(zones)} zones")
    for result in results:
        print(f"  - {result['zone']}: {result['hls_image_shape']}")
    print("=" * 70)

if __name__ == '__main__':
    main()
