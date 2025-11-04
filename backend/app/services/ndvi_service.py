"""
Servicio para calcular NDVI (Normalized Difference Vegetation Index)
"""
import numpy as np


class NDVIService:
    """
    Servicio para calcular índices de vegetación
    """

    @staticmethod
    def calculate_ndvi(red_band, nir_band, cloud_mask=None):
        """
        Calcular NDVI: (NIR - Red) / (NIR + Red)

        Args:
            red_band: Array NumPy con la banda Red (B04)
            nir_band: Array NumPy con la banda NIR (B08)
            cloud_mask: Máscara de nubes (opcional)

        Returns:
            Dict con ndvi, ndvi_masked, y estadísticas
        """
        # Evitar división por cero
        denominator = nir_band + red_band
        denominator[denominator == 0] = 0.0001

        # Calcular NDVI
        ndvi = (nir_band - red_band) / denominator

        # Aplicar máscara de nubes si existe
        if cloud_mask is not None:
            ndvi_masked = np.ma.masked_array(ndvi, mask=cloud_mask)
        else:
            ndvi_masked = ndvi

        # Normalizar NDVI a rango [-1, 1]
        ndvi_masked = np.clip(ndvi_masked, -1, 1)

        # Calcular estadísticas
        statistics = {
            'mean': float(np.ma.mean(ndvi_masked)),
            'std': float(np.ma.std(ndvi_masked)),
            'min': float(np.ma.min(ndvi_masked)),
            'max': float(np.ma.max(ndvi_masked)),
            'median': float(np.ma.median(ndvi_masked)),
            'cloud_coverage': float(np.sum(cloud_mask) / cloud_mask.size * 100) if cloud_mask is not None else 0
        }

        return {
            'ndvi': ndvi,
            'ndvi_masked': ndvi_masked,
            'statistics': statistics
        }

    @staticmethod
    def classify_vegetation_health(ndvi):
        """
        Clasificar salud de vegetación basado en valores NDVI

        Ranges:
        - NDVI < 0.2: Sin vegetación o muy poco saludable
        - 0.2 <= NDVI < 0.4: Vegetación poco saludable o con estrés
        - 0.4 <= NDVI < 0.6: Vegetación moderadamente saludable
        - NDVI >= 0.6: Vegetación saludable

        Args:
            ndvi: Array NumPy con valores NDVI

        Returns:
            Array con clasificación (0=sin vegetación, 1=estrés alto, 2=estrés medio, 3=saludable)
        """
        classification = np.zeros_like(ndvi, dtype=int)

        classification[ndvi < 0.2] = 0  # Sin vegetación
        classification[(ndvi >= 0.2) & (ndvi < 0.4)] = 1  # Estrés alto
        classification[(ndvi >= 0.4) & (ndvi < 0.6)] = 2  # Estrés medio
        classification[ndvi >= 0.6] = 3  # Saludable

        return classification

    @staticmethod
    def get_stress_mask(ndvi, threshold=0.4):
        """
        Obtener máscara binaria de áreas con estrés vegetal

        Args:
            ndvi: Array NumPy con valores NDVI
            threshold: Umbral para considerar estrés (default: 0.4)

        Returns:
            Máscara binaria donde True = estrés, False = saludable
        """
        return ndvi < threshold
