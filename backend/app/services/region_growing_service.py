"""
Servicio principal para análisis de estrés vegetal usando Region Growing
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


class RegionGrowingService:
    """
    Servicio que coordina el análisis completo:
    1. Obtención de imágenes satelitales
    2. Cálculo de NDVI
    3. Aplicación de Region Growing
    4. Conversión a coordenadas geográficas
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
        Analiza estrés vegetal en la región especificada

        Args:
            bbox: Bounding box con min_lat, min_lon, max_lat, max_lon
            date_from: Fecha inicial para la búsqueda (YYYY-MM-DD)
            date_to: Fecha final para la búsqueda (YYYY-MM-DD)

        Returns:
            Dict con GeoJSON de regiones y estadísticas
        """

        try:
            # 1. Obtener datos de Sentinel-2
            print(f"[1/4] Obteniendo imagen Sentinel-2 para bbox: {bbox}")
            sentinel_data = self.sentinel_service.get_sentinel2_data(
                bbox,
                date_from,
                date_to
            )

            red_band = sentinel_data['red']
            nir_band = sentinel_data['nir']
            cloud_mask = sentinel_data['cloud_mask']
            image_shape = red_band.shape

            print(f"      Imagen obtenida: {image_shape}, Nubes: {np.sum(cloud_mask)/cloud_mask.size*100:.1f}%")

            # 2. Calcular NDVI
            print("[2/4] Calculando NDVI...")
            ndvi_result = self.ndvi_service.calculate_ndvi(red_band, nir_band, cloud_mask)
            ndvi = ndvi_result['ndvi_masked']
            ndvi_stats = ndvi_result['statistics']

            print(f"      NDVI medio: {ndvi_stats['mean']:.3f}, rango: [{ndvi_stats['min']:.3f}, {ndvi_stats['max']:.3f}]")

            # 3. Aplicar Region Growing
            print("[3/4] Aplicando Region Growing...")
            # Usar -999 como valor especial para áreas enmascaradas (nubes)
            # El algoritmo debe ignorar estos píxeles
            ndvi_for_rg = np.ma.filled(ndvi, fill_value=-999)
            labeled_image, num_regions, regions_info = self.region_growing.region_growing(
                ndvi_for_rg
            )

            print(f"      Regiones detectadas: {num_regions}")

            # Clasificar regiones por nivel de estrés
            classified_regions = self.region_growing.classify_regions_by_stress(regions_info)

            print(f"      Alto estrés: {len(classified_regions['high_stress'])}, "
                  f"Medio: {len(classified_regions['medium_stress'])}, "
                  f"Bajo: {len(classified_regions['low_stress'])}")

            # 4. Convertir a coordenadas geográficas y GeoJSON
            print("[4/4] Convirtiendo a GeoJSON...")
            geojson = self.geo_converter.regions_to_geojson(
                regions_info,
                bbox,
                image_shape
            )

            # Calcular estadísticas
            statistics = self.geo_converter.calculate_statistics(
                regions_info,
                classified_regions,
                image_shape,
                resolution=10  # 10m para Sentinel-2
            )

            # Agregar información de la consulta
            statistics['date_from'] = sentinel_data['date_from']
            statistics['date_to'] = sentinel_data['date_to']
            statistics['cloud_coverage'] = ndvi_stats['cloud_coverage']

            print(f"      Total área: {statistics['total_area']:.2f} ha")
            print(f"      Estrés alto: {statistics['high_stress_area']:.2f} ha")

            # Crear imagen NDVI visualizable
            ndvi_image_base64 = self._create_ndvi_visualization(ndvi, image_shape)

            result = {
                'geojson': geojson,
                'statistics': statistics,
                'images': {
                    'rgb': sentinel_data.get('rgb_image_base64'),  # Imagen satelital real
                    'ndvi': ndvi_image_base64  # Mapa NDVI coloreado
                }
            }

            return result

        except Exception as e:
            print(f"Error en análisis: {str(e)}")
            raise Exception(f"Error al analizar la región: {str(e)}")

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
