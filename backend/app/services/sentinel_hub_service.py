"""
FastAPI service wrapper for Sentinel-2 downloads.
Uses pure functions from src.utils for core logic.
"""
from src.utils.sentinel_download import (
    create_sentinel_config,
    download_sentinel2_bands,
    create_cloud_mask,
    test_sentinel_connection
)
from src.utils.image_processing import (
    create_rgb_image,
    array_to_base64
)
from config.config import Settings
from app.utils import get_logger

logger = get_logger(__name__)


class SentinelHubService:
    """
    FastAPI service for Sentinel-2 imagery.
    Thin wrapper around src.utils functions.
    """

    def __init__(self):
        """Initialize Sentinel Hub configuration"""
        self.settings = Settings()

        # Use pure function from src.utils
        self.sh_config = create_sentinel_config(
            self.settings.sentinel_hub_client_id,
            self.settings.sentinel_hub_client_secret
        )

        # Resolution in meters per pixel (10m for Sentinel-2)
        self.resolution = 10

    def get_sentinel2_data(self, bbox_coords, date_from=None, date_to=None):
        """
        Get Sentinel-2 data (FastAPI wrapper).

        This method adds logging and error handling to the pure functions.

        Args:
            bbox_coords: Dict with min_lat, min_lon, max_lat, max_lon
            date_from: Start date (string YYYY-MM-DD)
            date_to: End date (string YYYY-MM-DD)

        Returns:
            Dict with Red, NIR bands and metadata
        """
        try:
            logger.info("Downloading Sentinel-2 data for bbox: %s", bbox_coords)

            # Use pure function from src.utils
            result = download_sentinel2_bands(
                bbox_coords=bbox_coords,
                config=self.sh_config,
                bands=['B02', 'B03', 'B04', 'B08', 'SCL'],
                date_from=date_from,
                date_to=date_to,
                resolution=self.resolution,
                max_cloud_coverage=0.5
            )

            # Process bands
            bands = result['bands']

            # Create cloud mask
            cloud_mask = create_cloud_mask(bands['SCL'])

            # Debug: View band value ranges
            logger.debug("RGB bands - Red: min=%.4f, max=%.4f, mean=%.4f",
                        bands['B04'].min(), bands['B04'].max(), bands['B04'].mean())
            logger.debug("RGB bands - Green: min=%.4f, max=%.4f, mean=%.4f",
                        bands['B03'].min(), bands['B03'].max(), bands['B03'].mean())
            logger.debug("RGB bands - Blue: min=%.4f, max=%.4f, mean=%.4f",
                        bands['B02'].min(), bands['B02'].max(), bands['B02'].mean())

            # Create RGB visualization using src.utils function
            rgb_array = create_rgb_image(
                bands['B04'],  # Red
                bands['B03'],  # Green
                bands['B02'],  # Blue
                gamma=0.8
            )

            logger.debug("Final RGB image: min=%d, max=%d, mean=%.2f",
                        rgb_array.min(), rgb_array.max(), rgb_array.mean())

            # Convert to base64
            rgb_base64 = array_to_base64(rgb_array, format='PNG')

            logger.info("Download successful: shape=%s", bands['B04'].shape)

            return {
                'red': bands['B04'],
                'green': bands['B03'],
                'blue': bands['B02'],
                'nir': bands['B08'],
                'cloud_mask': cloud_mask,
                'rgb_image_base64': rgb_base64,
                **result['metadata']
            }

        except Exception as e:
            logger.error("Error downloading Sentinel-2: %s", str(e), exc_info=True)
            raise Exception(f"Error obtaining data from Sentinel Hub: {str(e)}")

    def test_connection(self):
        """
        Test connection with Sentinel Hub.

        Returns:
            Dict with connection status
        """
        logger.info("Testing Sentinel Hub connection...")
        result = test_sentinel_connection(self.sh_config)

        if result['status'] == 'success':
            logger.info("Sentinel Hub connection successful")
        else:
            logger.error("Sentinel Hub connection failed: %s", result['message'])

        return result
