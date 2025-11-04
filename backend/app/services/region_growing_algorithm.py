"""
Implementación del algoritmo Region Growing para segmentación de imágenes NDVI
"""
import numpy as np
from collections import deque
import cv2


class RegionGrowingAlgorithm:
    """
    Algoritmo de Region Growing para segmentar áreas con características similares
    """

    def __init__(self, threshold=0.1, min_region_size=50):
        """
        Args:
            threshold: Umbral de similitud para agregar píxeles a la región
            min_region_size: Tamaño mínimo de región en píxeles
        """
        self.threshold = threshold
        self.min_region_size = min_region_size

    def region_growing(self, image, seeds=None):
        """
        Aplicar Region Growing sobre una imagen

        Args:
            image: Array 2D NumPy con valores NDVI
            seeds: Lista de puntos semilla [(y, x), ...]. Si None, se generan automáticamente

        Returns:
            labeled_image: Array con etiquetas de regiones
            num_regions: Número de regiones encontradas
            regions_info: Información de cada región
        """
        h, w = image.shape
        labeled = np.zeros((h, w), dtype=int)
        region_id = 0
        regions_info = []

        # Si no hay semillas, generar automáticamente
        if seeds is None:
            seeds = self._generate_seeds(image)

        # Para cada semilla, crecer región
        for seed_y, seed_x in seeds:
            if labeled[seed_y, seed_x] != 0:
                # Ya fue procesado
                continue

            # Crecer región desde esta semilla
            region = self._grow_region(image, seed_y, seed_x, labeled, region_id + 1)

            if len(region) >= self.min_region_size:
                region_id += 1

                # Calcular información de la región
                region_pixels = [image[y, x] for y, x in region]
                region_info = {
                    'id': region_id,
                    'size': len(region),
                    'mean_ndvi': float(np.mean(region_pixels)),
                    'std_ndvi': float(np.std(region_pixels)),
                    'min_ndvi': float(np.min(region_pixels)),
                    'max_ndvi': float(np.max(region_pixels)),
                    'pixels': region
                }
                regions_info.append(region_info)
            else:
                # Región muy pequeña, no la contamos
                for y, x in region:
                    labeled[y, x] = 0

        return labeled, region_id, regions_info

    def _grow_region(self, image, start_y, start_x, labeled, region_id):
        """
        Crecer una región desde un punto semilla usando BFS

        Args:
            image: Imagen NDVI
            start_y, start_x: Coordenadas del punto semilla
            labeled: Array de etiquetas
            region_id: ID de la región actual

        Returns:
            Lista de píxeles (y, x) en la región
        """
        h, w = image.shape
        seed_value = image[start_y, start_x]

        # Queue para BFS
        queue = deque([(start_y, start_x)])
        region = []
        visited = set()

        while queue:
            y, x = queue.popleft()

            # Verificar si ya fue visitado
            if (y, x) in visited:
                continue

            # Verificar límites
            if y < 0 or y >= h or x < 0 or x >= w:
                continue

            # Verificar si ya tiene etiqueta
            if labeled[y, x] != 0:
                continue

            # Ignorar píxeles enmascarados (nubes, valor -999)
            pixel_value = image[y, x]
            if pixel_value < -900:
                continue

            # Verificar similitud
            if abs(pixel_value - seed_value) > self.threshold:
                continue

            # Agregar a la región
            visited.add((y, x))
            region.append((y, x))
            labeled[y, x] = region_id

            # Agregar vecinos (4-conectividad)
            queue.append((y + 1, x))
            queue.append((y - 1, x))
            queue.append((y, x + 1))
            queue.append((y, x - 1))

        return region

    def _generate_seeds(self, image, grid_size=20):
        """
        Generar puntos semilla automáticamente en una cuadrícula

        Args:
            image: Imagen NDVI
            grid_size: Separación entre semillas en píxeles

        Returns:
            Lista de puntos semilla [(y, x), ...]
        """
        h, w = image.shape
        seeds = []

        for y in range(grid_size // 2, h, grid_size):
            for x in range(grid_size // 2, w, grid_size):
                # Verificar que el píxel tenga un valor válido
                # Ignorar píxeles enmascarados (nubes) que tienen valor -999
                if not np.isnan(image[y, x]) and not np.isinf(image[y, x]) and image[y, x] > -900:
                    seeds.append((y, x))

        return seeds

    def classify_regions_by_stress(self, regions_info):
        """
        Clasificar regiones por nivel de estrés basado en NDVI medio

        Args:
            regions_info: Lista de información de regiones

        Returns:
            Dict con regiones clasificadas por nivel de estrés
        """
        classified = {
            'high_stress': [],    # NDVI < 0.3
            'medium_stress': [],  # 0.3 <= NDVI < 0.5
            'low_stress': []      # NDVI >= 0.5
        }

        for region in regions_info:
            mean_ndvi = region['mean_ndvi']

            if mean_ndvi < 0.3:
                region['stress_level'] = 'high'
                classified['high_stress'].append(region)
            elif mean_ndvi < 0.5:
                region['stress_level'] = 'medium'
                classified['medium_stress'].append(region)
            else:
                region['stress_level'] = 'low'
                classified['low_stress'].append(region)

        return classified

    def merge_small_regions(self, labeled_image, regions_info, min_size=100):
        """
        Fusionar regiones pequeñas con sus vecinos más similares

        Args:
            labeled_image: Array con etiquetas de regiones
            regions_info: Información de las regiones
            min_size: Tamaño mínimo para no fusionar

        Returns:
            labeled_image actualizado, regions_info actualizado
        """
        # Por ahora, devolvemos sin cambios
        # Esto es una optimización que puede agregarse después
        return labeled_image, regions_info
