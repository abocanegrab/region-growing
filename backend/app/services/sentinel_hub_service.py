"""
Service to connect with Sentinel Hub API and obtain satellite imagery
"""
from sentinelhub import (
    SHConfig,
    BBox,
    CRS,
    DataCollection,
    SentinelHubRequest,
    MimeType,
    bbox_to_dimensions
)
from datetime import datetime, timedelta
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from config.config import Settings
from app.utils import get_logger

logger = get_logger(__name__)


class SentinelHubService:
    """
    Service to obtain Sentinel-2 images from Sentinel Hub
    """

    def __init__(self):
        """Initialize Sentinel Hub configuration"""
        self.settings = Settings()

        self.sh_config = SHConfig()
        self.sh_config.sh_client_id = self.settings.sentinel_hub_client_id
        self.sh_config.sh_client_secret = self.settings.sentinel_hub_client_secret

        # Resolución en metros por píxel (10m para Sentinel-2)
        self.resolution = 10

    def get_sentinel2_data(self, bbox_coords, date_from=None, date_to=None):
        """
        Obtener datos de Sentinel-2 para el área especificada

        Args:
            bbox_coords: Dict con min_lat, min_lon, max_lat, max_lon
            date_from: Fecha inicio (string YYYY-MM-DD)
            date_to: Fecha fin (string YYYY-MM-DD)

        Returns:
            Dict con las bandas Red, NIR y metadatos
        """
        # Configurar fechas
        if not date_to:
            date_to = datetime.now()
        else:
            date_to = datetime.strptime(date_to, '%Y-%m-%d')

        if not date_from:
            date_from = date_to - timedelta(days=30)
        else:
            date_from = datetime.strptime(date_from, '%Y-%m-%d')

        # Crear BBox
        bbox = BBox(
            bbox=[
                bbox_coords['min_lon'],
                bbox_coords['min_lat'],
                bbox_coords['max_lon'],
                bbox_coords['max_lat']
            ],
            crs=CRS.WGS84
        )

        # Calcular dimensiones de la imagen basado en resolución
        bbox_size = bbox_to_dimensions(bbox, resolution=self.resolution)

        # Sentinel Hub tiene límite de 2500x2500 píxeles
        # Si excede, ajustar proporcionalmente
        max_dimension = 2500
        width, height = bbox_size

        if width > max_dimension or height > max_dimension:
            # Calculate scale factor
            scale = min(max_dimension / width, max_dimension / height)
            bbox_size = (int(width * scale), int(height * scale))
            logger.info("Image too large (%dx%d), reduced to %s", width, height, bbox_size)

        # Evalscript para obtener bandas B02 (Blue), B03 (Green), B04 (Red), B08 (NIR), SCL
        evalscript = """
        //VERSION=3
        function setup() {
            return {
                input: [{
                    bands: ["B02", "B03", "B04", "B08", "SCL"],
                    units: "DN"
                }],
                output: {
                    bands: 5,
                    sampleType: "FLOAT32"
                }
            };
        }

        function evaluatePixel(sample) {
            return [sample.B02, sample.B03, sample.B04, sample.B08, sample.SCL];
        }
        """

        # Crear request
        request = SentinelHubRequest(
            evalscript=evalscript,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L2A,
                    time_interval=(date_from, date_to),
                    maxcc=0.5  # Máximo 50% de cobertura de nubes
                )
            ],
            responses=[
                SentinelHubRequest.output_response('default', MimeType.TIFF)
            ],
            bbox=bbox,
            size=bbox_size,
            config=self.sh_config
        )

        try:
            # Obtener datos
            data = request.get_data()[0]

            # Separar bandas
            blue_band = data[:, :, 0]
            green_band = data[:, :, 1]
            red_band = data[:, :, 2]
            nir_band = data[:, :, 3]
            scl_band = data[:, :, 4]

            # Crear máscara de nubes (SCL band)
            # SCL: 3=cloud shadows, 8=cloud medium prob, 9=cloud high prob, 10=thin cirrus
            cloud_mask = np.isin(scl_band, [3, 8, 9, 10])

            # Create RGB image for visualization
            # Debug: View band value ranges
            logger.debug("RGB bands - Red: min=%.4f, max=%.4f, mean=%.4f",
                        red_band.min(), red_band.max(), red_band.mean())
            logger.debug("RGB bands - Green: min=%.4f, max=%.4f, mean=%.4f",
                        green_band.min(), green_band.max(), green_band.mean())
            logger.debug("RGB bands - Blue: min=%.4f, max=%.4f, mean=%.4f",
                        blue_band.min(), blue_band.max(), blue_band.mean())

            # Normalize RGB in a simple and robust way
            # Sentinel-2 DN has typical values between 0-10000
            rgb_image = np.stack([red_band, green_band, blue_band], axis=2)

            logger.debug("Band ranges - R:[%.0f,%.0f] G:[%.0f,%.0f] B:[%.0f,%.0f]",
                        red_band.min(), red_band.max(),
                        green_band.min(), green_band.max(),
                        blue_band.min(), blue_band.max())

            # Use percentiles for robust normalization (avoids extreme values)
            p2, p98 = np.percentile(rgb_image, [2, 98])
            logger.debug("RGB percentiles - P2:%.0f, P98:%.0f", p2, p98)

            # Normalizar a 0-1 usando percentiles
            rgb_normalized = (rgb_image - p2) / (p98 - p2 + 1e-10)  # +epsilon para evitar división por 0
            rgb_normalized = np.clip(rgb_normalized, 0, 1)

            # Ajuste gamma para mejorar contraste (gamma < 1 aclara, gamma > 1 oscurece)
            gamma = 0.8  # Aclara ligeramente la imagen
            rgb_normalized = np.power(rgb_normalized, gamma)

            # Convert to uint8
            rgb_image = (rgb_normalized * 255).astype(np.uint8)

            logger.debug("Final RGB image: min=%d, max=%d, mean=%.2f",
                        rgb_image.min(), rgb_image.max(), rgb_image.mean())

            # Convertir imagen RGB a base64 para enviar al frontend
            rgb_image_pil = Image.fromarray(rgb_image)
            buffered = BytesIO()
            rgb_image_pil.save(buffered, format="PNG")
            rgb_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

            return {
                'red': red_band,
                'nir': nir_band,
                'cloud_mask': cloud_mask,
                'rgb_image_base64': rgb_base64,  # Imagen RGB en base64 para el frontend
                'bbox': bbox,
                'bbox_coords': bbox_coords,
                'dimensions': bbox_size,
                'date_from': date_from.strftime('%Y-%m-%d'),
                'date_to': date_to.strftime('%Y-%m-%d')
            }

        except Exception as e:
            logger.error("Error obtaining data from Sentinel Hub: %s", str(e), exc_info=True)
            raise Exception(f"Error obtaining data from Sentinel Hub: {str(e)}")

    def test_connection(self):
        """
        Test connection with Sentinel Hub

        Returns:
            Dict with connection status
        """
        try:
            logger.info("Testing Sentinel Hub connection...")
            # Try a simple request to verify credentials
            test_bbox = BBox(bbox=[-77.1, -12.1, -77.0, -12.0], crs=CRS.WGS84)

            evalscript = """
            //VERSION=3
            function setup() {
                return {
                    input: ["B04"],
                    output: { bands: 1 }
                };
            }
            function evaluatePixel(sample) {
                return [sample.B04];
            }
            """

            request = SentinelHubRequest(
                evalscript=evalscript,
                input_data=[
                    SentinelHubRequest.input_data(
                        data_collection=DataCollection.SENTINEL2_L2A,
                        time_interval=(datetime.now() - timedelta(days=30), datetime.now())
                    )
                ],
                responses=[
                    SentinelHubRequest.output_response('default', MimeType.TIFF)
                ],
                bbox=test_bbox,
                size=(100, 100),
                config=self.sh_config
            )

            # Try to obtain data
            data = request.get_data()

            logger.info("Sentinel Hub connection successful")
            return {
                'status': 'success',
                'message': 'Successful connection to Sentinel Hub',
                'data_shape': data[0].shape if data else None
            }

        except Exception as e:
            logger.error("Sentinel Hub connection failed: %s", str(e), exc_info=True)
            return {
                'status': 'error',
                'message': f'Connection error: {str(e)}'
            }
