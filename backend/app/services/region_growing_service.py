"""
Service layer for Region Growing analysis workflow orchestration.

This service coordinates the complete vegetation stress analysis workflow,
acting as a thin wrapper that orchestrates calls to pure functions in src/.
"""
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from typing import Dict, Optional

# Import pure functions from src/
from src.algorithms.classic_region_growing import ClassicRegionGrowing
from src.features.ndvi_calculator import calculate_ndvi

# Import backend-specific services
from app.services.sentinel_hub_service import SentinelHubService
from app.services.geo_converter_service import GeoConverterService
from app.utils import get_logger

logger = get_logger(__name__)


class RegionGrowingService:
    """
    Orchestrates the complete Region Growing analysis workflow.

    This service is a thin wrapper that coordinates:
    1. Data acquisition (Sentinel-2 via SentinelHubService)
    2. NDVI calculation (via src.features.ndvi_calculator)
    3. Region Growing segmentation (via src.algorithms.ClassicRegionGrowing)
    4. Geographic conversion (via GeoConverterService)
    5. Visualization generation

    Note: Business logic is in src/, this service only orchestrates and handles I/O.
    """

    def __init__(self):
        """Initialize service with dependencies."""
        self.sentinel_service = SentinelHubService()
        self.algorithm = ClassicRegionGrowing(
            threshold=0.1,
            min_region_size=50,
            cloud_mask_value=-999.0
        )
        self.geo_converter = GeoConverterService()

    def analyze_stress(
        self,
        bbox: Dict[str, float],
        date_from: Optional[str] = None,
        date_to: Optional[str] = None
    ) -> Dict:
        """
        Analyze vegetation stress in the specified region.

        This method orchestrates the complete workflow without containing
        business logic. All processing is delegated to modules in src/.

        Args:
            bbox: Bounding box with min_lat, min_lon, max_lat, max_lon
            date_from: Start date for search (YYYY-MM-DD)
            date_to: End date for search (YYYY-MM-DD)

        Returns:
            Dict with:
            - geojson: GeoJSON FeatureCollection with regions
            - statistics: Analysis statistics
            - regions: List of regions with stress classification
            - images: Base64-encoded visualization images

        Raises:
            Exception: If any step in the workflow fails
        """
        try:
            # Step 1: Acquire satellite data
            logger.info("[1/4] Obtaining Sentinel-2 image for bbox: %s", bbox)
            sentinel_data = self.sentinel_service.get_sentinel2_data(
                bbox,
                date_from,
                date_to
            )

            red_band = sentinel_data['red']
            nir_band = sentinel_data['nir']
            green_band = sentinel_data.get('green')
            cloud_mask = sentinel_data['cloud_mask']
            image_shape = red_band.shape

            cloud_percentage = np.sum(cloud_mask) / cloud_mask.size * 100
            logger.info("Image obtained: shape=%s, cloud_coverage=%.1f%%",
                       image_shape, cloud_percentage)

            # Step 2: Calculate NDVI using pure function from src/
            logger.info("[2/4] Calculating NDVI...")
            ndvi_result = calculate_ndvi(red_band, nir_band, cloud_mask)
            ndvi = ndvi_result['ndvi_masked']
            ndvi_stats = ndvi_result['statistics']

            logger.info("NDVI calculated: mean=%.3f, range=[%.3f, %.3f]",
                       ndvi_stats['mean'], ndvi_stats['min'], ndvi_stats['max'])

            # Step 3: Apply Region Growing algorithm from src/
            logger.info("[3/4] Applying Region Growing algorithm...")
            ndvi_for_rg = np.ma.filled(ndvi, fill_value=-999.0)
            labeled_image, num_regions, regions_info = self.algorithm.segment(ndvi_for_rg)

            logger.info("Regions detected: %d", num_regions)

            # Classify regions by stress level using algorithm method
            classified_regions = self.algorithm.classify_by_stress(regions_info)

            logger.info("Stress classification: high=%d, medium=%d, low=%d",
                       len(classified_regions['high_stress']),
                       len(classified_regions['medium_stress']),
                       len(classified_regions['low_stress']))

            # Step 4: Convert to geographic coordinates and GeoJSON
            logger.info("[4/4] Converting to GeoJSON...")
            geojson = self.geo_converter.regions_to_geojson(
                regions_info,
                bbox,
                image_shape
            )

            # Calculate statistics
            statistics = self.geo_converter.calculate_statistics(
                regions_info,
                classified_regions,
                image_shape,
                resolution=10  # 10m for Sentinel-2
            )

            # Add query metadata
            statistics['date_from'] = sentinel_data['date_from']
            statistics['date_to'] = sentinel_data['date_to']
            statistics['cloud_coverage'] = ndvi_stats['cloud_coverage']

            logger.info("Analysis completed: total_area=%.2f ha, high_stress_area=%.2f ha",
                       statistics['total_area'], statistics['high_stress_area'])

            # Generate visualizations
            ndvi_image_base64 = self._create_ndvi_visualization(ndvi, image_shape)
            false_color_base64 = self._create_false_color_image(
                sentinel_data.get('nir'),
                sentinel_data.get('red'),
                sentinel_data.get('green')
            )

            # Prepare regions list for frontend
            regions_list = self._prepare_regions_list(regions_info)

            result = {
                'geojson': geojson,
                'statistics': statistics,
                'regions': regions_list,
                'images': {
                    'rgb': sentinel_data.get('rgb_image_base64'),
                    'ndvi': ndvi_image_base64,
                    'false_color': false_color_base64
                }
            }

            return result

        except Exception as e:
            logger.error("Error during analysis: %s", str(e), exc_info=True)
            raise Exception(f"Error analyzing region: {str(e)}")

    def test_sentinel_connection(self) -> Dict:
        """
        Test connection to Sentinel Hub API.

        Returns:
            Dict with connection status
        """
        return self.sentinel_service.test_connection()

    def _prepare_regions_list(self, regions_info: list) -> list:
        """
        Prepare regions list for frontend consumption.

        Args:
            regions_info: List of region dicts from algorithm

        Returns:
            List of simplified region dicts with calculated areas
        """
        regions_list = []
        pixel_area_m2 = 10 * 10  # 10m resolution

        for region in regions_info:
            area_ha = (region['size'] * pixel_area_m2) / 10000

            regions_list.append({
                'id': region['id'],
                'stress_level': region.get('stress_level', 'unknown'),
                'ndvi_mean': round(region['mean_ndvi'], 3),
                'area': round(area_ha, 2)
            })

        return regions_list

    def _create_ndvi_visualization(
        self,
        ndvi: np.ndarray,
        image_shape: tuple
    ) -> str:
        """
        Create colored NDVI visualization image.

        Generates a false-color image with gradient: Red → Yellow → Green
        representing NDVI values from low to high.

        Args:
            ndvi: NDVI array (may be masked array)
            image_shape: Shape of the original image

        Returns:
            Base64-encoded PNG image string
        """
        # Convert masked array to regular array
        if np.ma.is_masked(ndvi):
            ndvi_array = np.ma.filled(ndvi, fill_value=-1)
            cloud_mask = ndvi.mask
        else:
            ndvi_array = ndvi
            cloud_mask = None

        # Normalize NDVI to [0, 1] range
        ndvi_normalized = (ndvi_array + 1) / 2
        ndvi_normalized = np.clip(ndvi_normalized, 0, 1)

        # Create RGB image with gradient: Red → Yellow → Green
        h, w = ndvi_array.shape
        ndvi_colored = np.zeros((h, w, 3), dtype=np.uint8)

        # Split at midpoint for two-stage gradient
        mask_low = ndvi_normalized < 0.5
        mask_high = ~mask_low

        # First half: Red (255,0,0) → Yellow (255,255,0)
        t_low = ndvi_normalized[mask_low] * 2
        ndvi_colored[mask_low, 0] = 255  # Red constant
        ndvi_colored[mask_low, 1] = (t_low * 255).astype(np.uint8)  # Green increases
        ndvi_colored[mask_low, 2] = 0  # Blue constant

        # Second half: Yellow (255,255,0) → Green (0,255,0)
        t_high = (ndvi_normalized[mask_high] - 0.5) * 2
        ndvi_colored[mask_high, 0] = ((1 - t_high) * 255).astype(np.uint8)  # Red decreases
        ndvi_colored[mask_high, 1] = 255  # Green constant
        ndvi_colored[mask_high, 2] = 0  # Blue constant

        # Mark cloud pixels as gray
        if cloud_mask is not None:
            ndvi_colored[cloud_mask] = [128, 128, 128]

        # Convert to base64
        ndvi_image_pil = Image.fromarray(ndvi_colored)
        buffered = BytesIO()
        ndvi_image_pil.save(buffered, format="PNG")
        ndvi_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return ndvi_base64

    def _create_false_color_image(
        self,
        nir_band: Optional[np.ndarray],
        red_band: Optional[np.ndarray],
        green_band: Optional[np.ndarray]
    ) -> Optional[str]:
        """
        Create false color composite image (NIR-Red-Green).

        This composition (NIR → R, Red → G, Green → B) highlights vegetation
        in red/pink tones, making it easy to visually identify vegetated areas.

        Args:
            nir_band: Near-infrared band (B08)
            red_band: Red band (B04)
            green_band: Green band (B03)

        Returns:
            Base64-encoded PNG image string, or None if bands missing
        """
        if nir_band is None or red_band is None or green_band is None:
            logger.warning("Missing bands for false color image")
            return None

        # Stack bands: NIR → R, Red → G, Green → B
        false_color = np.stack([nir_band, red_band, green_band], axis=2)

        # Robust normalization using percentiles
        p2, p98 = np.percentile(false_color, [2, 98])
        logger.debug("False color percentiles - P2:%.0f, P98:%.0f", p2, p98)

        false_color_normalized = (false_color - p2) / (p98 - p2 + 1e-10)
        false_color_normalized = np.clip(false_color_normalized, 0, 1)

        # Gamma adjustment for better contrast
        gamma = 0.8
        false_color_normalized = np.power(false_color_normalized, gamma)

        # Convert to uint8
        false_color_image = (false_color_normalized * 255).astype(np.uint8)

        logger.debug("False color image: min=%d, max=%d, mean=%.2f",
                    false_color_image.min(), false_color_image.max(),
                    false_color_image.mean())

        # Convert to base64
        false_color_pil = Image.fromarray(false_color_image)
        buffered = BytesIO()
        false_color_pil.save(buffered, format="PNG")
        false_color_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return false_color_base64
