"""
Diagnose why Sentinel Hub is returning empty data.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.sentinel_download import create_sentinel_config, test_sentinel_connection
import os

def main():
    print("Diagnosing Sentinel Hub connection...")
    print("=" * 70)
    
    # Check credentials
    client_id = os.getenv('SENTINELHUB_CLIENT_ID')
    client_secret = os.getenv('SENTINELHUB_CLIENT_SECRET')
    
    if not client_id or not client_secret:
        credentials_file = Path(__file__).parent.parent / 'sentinelhub-secrets_.txt'
        if credentials_file.exists():
            print(f"Loading credentials from {credentials_file}")
            with open(credentials_file, 'r') as f:
                lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                if len(lines) >= 2:
                    client_id = lines[0]
                    client_secret = lines[1]
    
    if not client_id or not client_secret:
        print("ERROR: No credentials found!")
        print("\nTo fix:")
        print("1. Set environment variables:")
        print("   export SENTINELHUB_CLIENT_ID='your_id'")
        print("   export SENTINELHUB_CLIENT_SECRET='your_secret'")
        print("\n2. Or create sentinelhub-secrets_.txt with:")
        print("   your_client_id")
        print("   your_client_secret")
        return
    
    print(f"Client ID: {client_id[:10]}...")
    print(f"Client Secret: {'*' * 20}")
    
    # Test connection
    print("\nTesting connection to Sentinel Hub...")
    config = create_sentinel_config(client_id, client_secret)
    result = test_sentinel_connection(config)
    
    print(f"\nStatus: {result['status']}")
    print(f"Message: {result['message']}")
    if result.get('data_shape'):
        print(f"Data shape: {result['data_shape']}")
    
    if result['status'] == 'success':
        print("\nConnection successful!")
        print("\nPossible reasons for empty data:")
        print("1. No Sentinel-2 data available for the specific date (2024-01-15)")
        print("2. 100% cloud coverage for that date")
        print("3. Area outside Sentinel-2 coverage")
        print("\nSuggestions:")
        print("- Try a different date range (e.g., last 30 days)")
        print("- Increase max_cloud_coverage parameter")
        print("- Verify the area has Sentinel-2 coverage")
    else:
        print("\nConnection failed!")
        print("Check your credentials and try again.")

if __name__ == '__main__':
    main()
