"""
Servicio para convertir regiones de píxeles a coordenadas geográficas
"""
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, mapping
from shapely.ops import unary_union
import cv2


class GeoConverterService:
    """
    Convierte regiones detectadas a coordenadas geográficas y GeoJSON
    """

    @staticmethod
    def pixels_to_latlon(pixel_coords, bbox_coords, image_shape):
        """
        Convertir coordenadas de píxeles a lat/lon

        Args:
            pixel_coords: Lista de (y, x) en píxeles
            bbox_coords: Dict con min_lat, min_lon, max_lat, max_lon
            image_shape: (height, width) de la imagen

        Returns:
            Lista de (lat, lon)
        """
        h, w = image_shape

        min_lat = bbox_coords['min_lat']
        max_lat = bbox_coords['max_lat']
        min_lon = bbox_coords['min_lon']
        max_lon = bbox_coords['max_lon']

        lat_per_pixel = (max_lat - min_lat) / h
        lon_per_pixel = (max_lon - min_lon) / w

        latlon_coords = []
        for y, x in pixel_coords:
            lat = max_lat - (y * lat_per_pixel)  # Invertido porque y=0 está arriba
            lon = min_lon + (x * lon_per_pixel)
            latlon_coords.append((lat, lon))

        return latlon_coords

    @staticmethod
    def region_to_polygon(region_pixels, bbox_coords, image_shape, simplify_tolerance=0.0001):
        """
        Convertir una región de píxeles a un polígono geográfico

        Args:
            region_pixels: Lista de (y, x) píxeles de la región
            bbox_coords: Dict con coordenadas del bbox
            image_shape: (height, width)
            simplify_tolerance: Tolerancia para simplificar polígono

        Returns:
            Polígono Shapely en coordenadas geográficas
        """
        # Crear imagen binaria de la región
        h, w = image_shape
        mask = np.zeros((h, w), dtype=np.uint8)

        for y, x in region_pixels:
            if 0 <= y < h and 0 <= x < w:
                mask[y, x] = 255

        # Encontrar contornos
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Usar el contorno más grande
        contour = max(contours, key=cv2.contourArea)

        # Convertir contorno a coordenadas geográficas
        pixel_coords = [(point[0][1], point[0][0]) for point in contour]  # (y, x)
        latlon_coords = GeoConverterService.pixels_to_latlon(
            pixel_coords, bbox_coords, image_shape
        )

        # Crear polígono (lon, lat para GeoJSON)
        polygon_coords = [(lon, lat) for lat, lon in latlon_coords]

        try:
            polygon = Polygon(polygon_coords)

            # Simplificar si es muy complejo
            if simplify_tolerance > 0:
                polygon = polygon.simplify(simplify_tolerance, preserve_topology=True)

            return polygon if polygon.is_valid else None
        except:
            return None

    @staticmethod
    def regions_to_geojson(regions_info, bbox_coords, image_shape):
        """
        Convertir lista de regiones a GeoJSON

        Args:
            regions_info: Lista con información de regiones (del Region Growing)
            bbox_coords: Dict con coordenadas del bbox
            image_shape: (height, width)

        Returns:
            Dict GeoJSON FeatureCollection
        """
        features = []

        for region in regions_info:
            # Convertir región a polígono
            polygon = GeoConverterService.region_to_polygon(
                region['pixels'],
                bbox_coords,
                image_shape
            )

            if polygon is None or polygon.is_empty:
                continue

            # Crear feature GeoJSON
            feature = {
                'type': 'Feature',
                'geometry': mapping(polygon),
                'properties': {
                    'region_id': region['id'],
                    'size': region['size'],
                    'mean_ndvi': round(region['mean_ndvi'], 3),
                    'std_ndvi': round(region['std_ndvi'], 3),
                    'min_ndvi': round(region['min_ndvi'], 3),
                    'max_ndvi': round(region['max_ndvi'], 3),
                    'stress_level': region.get('stress_level', 'unknown')
                }
            }

            features.append(feature)

        geojson = {
            'type': 'FeatureCollection',
            'features': features
        }

        return geojson

    @staticmethod
    def calculate_statistics(regions_info, classified_regions, image_shape, resolution=10):
        """
        Calcular estadísticas del análisis

        Args:
            regions_info: Lista con información de todas las regiones
            classified_regions: Dict con regiones clasificadas por estrés
            image_shape: (height, width)
            resolution: Resolución en metros por píxel (default: 10m para Sentinel-2)

        Returns:
            Dict con estadísticas
        """
        # Calcular área total en hectáreas
        total_pixels = image_shape[0] * image_shape[1]
        pixel_area_m2 = resolution * resolution  # m² por píxel
        total_area_ha = (total_pixels * pixel_area_m2) / 10000  # Convertir a hectáreas

        # Calcular áreas por nivel de estrés
        high_stress_pixels = sum(r['size'] for r in classified_regions.get('high_stress', []))
        medium_stress_pixels = sum(r['size'] for r in classified_regions.get('medium_stress', []))
        low_stress_pixels = sum(r['size'] for r in classified_regions.get('low_stress', []))

        high_stress_area_ha = (high_stress_pixels * pixel_area_m2) / 10000
        medium_stress_area_ha = (medium_stress_pixels * pixel_area_m2) / 10000
        low_stress_area_ha = (low_stress_pixels * pixel_area_m2) / 10000

        # NDVI promedio general
        if regions_info:
            mean_ndvi = np.mean([r['mean_ndvi'] for r in regions_info])
        else:
            mean_ndvi = 0

        statistics = {
            'total_area': round(total_area_ha, 2),
            'high_stress_area': round(high_stress_area_ha, 2),
            'medium_stress_area': round(medium_stress_area_ha, 2),
            'low_stress_area': round(low_stress_area_ha, 2),
            'mean_ndvi': round(float(mean_ndvi), 3),
            'num_regions': len(regions_info),
            'num_high_stress_regions': len(classified_regions.get('high_stress', [])),
            'num_medium_stress_regions': len(classified_regions.get('medium_stress', [])),
            'num_low_stress_regions': len(classified_regions.get('low_stress', []))
        }

        return statistics
