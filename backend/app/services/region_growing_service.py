"""
Main service for vegetation stress analysis using Region Growing algorithm
"""
import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image
from typing import Dict, Optional
from app.services.sentinel_hub_service import SentinelHubService
from app.services.ndvi_service import NDVIService
from app.services.region_growing_algorithm import RegionGrowingAlgorithm
from app.services.geo_converter_service import GeoConverterService
from app.utils import get_logger

logger = get_logger(__name__)


class RegionGrowingService:
    """
    Service that coordinates the complete analysis workflow:
    1. Obtaining satellite imagery
    2. Calculating NDVI
    3. Applying Region Growing algorithm
    4. Converting to geographic coordinates
    """

    def __init__(self):
        self.sentinel_service = SentinelHubService()
        self.ndvi_service = NDVIService()
        self.region_growing = RegionGrowingAlgorithm(threshold=0.1, min_region_size=50)
        self.geo_converter = GeoConverterService()

    def analyze_stress(
        self,
        bbox: Dict[str, float],
        date_from: Optional[str] = None,
        date_to: Optional[str] = None
    ) -> Dict:
        """
        Analyze vegetation stress in the specified region

        Args:
            bbox: Bounding box with min_lat, min_lon, max_lat, max_lon
            date_from: Start date for search (YYYY-MM-DD)
            date_to: End date for search (YYYY-MM-DD)

        Returns:
            Dict with GeoJSON of regions and statistics
        """

        try:
            # 1. Obtain Sentinel-2 data
            logger.info("[1/4] Obtaining Sentinel-2 image for bbox: %s", bbox)
            sentinel_data = self.sentinel_service.get_sentinel2_data(
                bbox,
                date_from,
                date_to
            )

            red_band = sentinel_data['red']
            nir_band = sentinel_data['nir']
            green_band = sentinel_data.get('green')  # Necesitamos green para falso color
            cloud_mask = sentinel_data['cloud_mask']
            image_shape = red_band.shape
            
            # Guardar green band en sentinel_data para uso posterior
            sentinel_data['green'] = green_band

            cloud_percentage = np.sum(cloud_mask) / cloud_mask.size * 100
            logger.info("Image obtained: shape=%s, cloud_coverage=%.1f%%",
                       image_shape, cloud_percentage)

            # 2. Calculate NDVI
            logger.info("[2/4] Calculating NDVI...")
            ndvi_result = self.ndvi_service.calculate_ndvi(red_band, nir_band, cloud_mask)
            ndvi = ndvi_result['ndvi_masked']
            ndvi_stats = ndvi_result['statistics']

            logger.info("NDVI calculated: mean=%.3f, range=[%.3f, %.3f]",
                       ndvi_stats['mean'], ndvi_stats['min'], ndvi_stats['max'])

            # 3. Apply Region Growing
            logger.info("[3/4] Applying Region Growing algorithm...")
            # Use -999 as special value for masked areas (clouds)
            # The algorithm should ignore these pixels
            ndvi_for_rg = np.ma.filled(ndvi, fill_value=-999)
            labeled_image, num_regions, regions_info = self.region_growing.region_growing(
                ndvi_for_rg
            )

            logger.info("Regions detected: %d", num_regions)

            # Classify regions by stress level
            classified_regions = self.region_growing.classify_regions_by_stress(regions_info)

            logger.info("Stress classification: high=%d, medium=%d, low=%d",
                       len(classified_regions['high_stress']),
                       len(classified_regions['medium_stress']),
                       len(classified_regions['low_stress']))

            # 4. Convert to geographic coordinates and GeoJSON
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

            # Add query information
            statistics['date_from'] = sentinel_data['date_from']
            statistics['date_to'] = sentinel_data['date_to']
            statistics['cloud_coverage'] = ndvi_stats['cloud_coverage']

            logger.info("Analysis completed: total_area=%.2f ha, high_stress_area=%.2f ha",
                       statistics['total_area'], statistics['high_stress_area'])

            # Crear imagen NDVI visualizable
            ndvi_image_base64 = self._create_ndvi_visualization(ndvi, image_shape)

            # Crear imagen de falso color (NIR-Red-Green)
            false_color_base64 = self._create_false_color_image(
                sentinel_data.get('nir'),
                sentinel_data.get('red'),
                sentinel_data.get('green')
            )

            # Preparar lista de regiones para el frontend
            regions_list = []
            for region in regions_info:
                # Calcular área en hectáreas
                pixel_area_m2 = 10 * 10  # 10m resolución
                area_ha = (region['size'] * pixel_area_m2) / 10000
                
                regions_list.append({
                    'id': region['id'],
                    'stress_level': region.get('stress_level', 'unknown'),
                    'ndvi_mean': round(region['mean_ndvi'], 3),
                    'area': round(area_ha, 2)
                })

            result = {
                'geojson': geojson,
                'statistics': statistics,
                'regions': regions_list,  # Lista de regiones para la tabla
                'images': {
                    'rgb': sentinel_data.get('rgb_image_base64'),  # Imagen satelital real
                    'ndvi': ndvi_image_base64,  # Mapa NDVI coloreado
                    'false_color': false_color_base64  # Falso color NIR-Red-Green
                }
            }

            return result

        except Exception as e:
            logger.error("Error during analysis: %s", str(e), exc_info=True)
            raise Exception(f"Error analyzing region: {str(e)}")

    def test_sentinel_connection(self):
        """
        Probar la conexión con Sentinel Hub

        Returns:
            Dict con status de la conexión
        """
        return self.sentinel_service.test_connection()

    def _create_ndvi_visualization(self, ndvi, image_shape):
        """
        Crear visualización coloreada del NDVI

        Args:
            ndvi: Array con valores NDVI (puede ser masked array)
            image_shape: Shape de la imagen

        Returns:
            String base64 de la imagen NDVI coloreada
        """
        # Convertir masked array a array normal
        if np.ma.is_masked(ndvi):
            ndvi_array = np.ma.filled(ndvi, fill_value=-1)
            cloud_mask = ndvi.mask
        else:
            ndvi_array = ndvi
            cloud_mask = None

        # Normalizar NDVI a rango 0-1
        ndvi_normalized = (ndvi_array + 1) / 2  # De [-1, 1] a [0, 1]
        ndvi_normalized = np.clip(ndvi_normalized, 0, 1)

        # Crear colormap personalizado: Rojo → Amarillo → Verde
        # Usando operaciones vectorizadas de NumPy para velocidad
        h, w = ndvi_array.shape
        ndvi_colored = np.zeros((h, w, 3), dtype=np.uint8)

        # Máscara para primera mitad (Rojo → Amarillo)
        mask_low = ndvi_normalized < 0.5
        # Máscara para segunda mitad (Amarillo → Verde)
        mask_high = ~mask_low

        # Primera mitad: Rojo (255,0,0) → Amarillo (255,255,0)
        t_low = ndvi_normalized[mask_low] * 2  # Normalizar a [0, 1]
        ndvi_colored[mask_low, 0] = 255  # R = 255
        ndvi_colored[mask_low, 1] = (t_low * 255).astype(np.uint8)  # G aumenta
        ndvi_colored[mask_low, 2] = 0  # B = 0

        # Segunda mitad: Amarillo (255,255,0) → Verde (0,255,0)
        t_high = (ndvi_normalized[mask_high] - 0.5) * 2  # Normalizar a [0, 1]
        ndvi_colored[mask_high, 0] = ((1 - t_high) * 255).astype(np.uint8)  # R disminuye
        ndvi_colored[mask_high, 1] = 255  # G = 255
        ndvi_colored[mask_high, 2] = 0  # B = 0

        # Marcar áreas enmascaradas en gris
        if cloud_mask is not None:
            ndvi_colored[cloud_mask] = [128, 128, 128]

        # Convertir a base64
        ndvi_image_pil = Image.fromarray(ndvi_colored)
        buffered = BytesIO()
        ndvi_image_pil.save(buffered, format="PNG")
        ndvi_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return ndvi_base64

    def _create_false_color_image(self, nir_band, red_band, green_band):
        """
        Crear imagen de falso color (NIR-Red-Green)
        
        Composición: NIR → R, Red → G, Green → B
        Esta composición resalta la vegetación en tonos rojos/rosados
        
        Args:
            nir_band: Banda NIR (B08)
            red_band: Banda Red (B04)
            green_band: Banda Green (B03)
            
        Returns:
            String base64 de la imagen de falso color
        """
        if nir_band is None or red_band is None or green_band is None:
            logger.warning("Missing bands for false color image")
            return None
            
        # Stack bands: NIR → R, Red → G, Green → B
        false_color = np.stack([nir_band, red_band, green_band], axis=2)
        
        # Normalización robusta usando percentiles
        p2, p98 = np.percentile(false_color, [2, 98])
        logger.debug("False color percentiles - P2:%.0f, P98:%.0f", p2, p98)
        
        # Normalizar a 0-1
        false_color_normalized = (false_color - p2) / (p98 - p2 + 1e-10)
        false_color_normalized = np.clip(false_color_normalized, 0, 1)
        
        # Ajuste gamma para mejorar contraste
        gamma = 0.8
        false_color_normalized = np.power(false_color_normalized, gamma)
        
        # Convertir a uint8
        false_color_image = (false_color_normalized * 255).astype(np.uint8)
        
        logger.debug("False color image: min=%d, max=%d, mean=%.2f",
                    false_color_image.min(), false_color_image.max(), false_color_image.mean())
        
        # Convertir a base64
        false_color_pil = Image.fromarray(false_color_image)
        buffered = BytesIO()
        false_color_pil.save(buffered, format="PNG")
        false_color_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return false_color_base64
