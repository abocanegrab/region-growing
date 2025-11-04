"""
Servicio para conectar con Sentinel Hub API y obtener imágenes satelitales
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
from config.config import get_config


class SentinelHubService:
    """
    Servicio para obtener imágenes de Sentinel-2 desde Sentinel Hub
    """

    def __init__(self):
        """Inicializar configuración de Sentinel Hub"""
        config = get_config()

        self.sh_config = SHConfig()
        self.sh_config.sh_client_id = config.SENTINEL_HUB_CLIENT_ID
        self.sh_config.sh_client_secret = config.SENTINEL_HUB_CLIENT_SECRET

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
            # Calcular factor de escala
            scale = min(max_dimension / width, max_dimension / height)
            bbox_size = (int(width * scale), int(height * scale))
            print(f"      Imagen muy grande ({width}x{height}), reducida a {bbox_size}")

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

            # Crear imagen RGB para visualización
            # Debug: Ver rango de valores de las bandas
            print(f"      DEBUG RGB - Red band: min={red_band.min():.4f}, max={red_band.max():.4f}, mean={red_band.mean():.4f}")
            print(f"      DEBUG RGB - Green band: min={green_band.min():.4f}, max={green_band.max():.4f}, mean={green_band.mean():.4f}")
            print(f"      DEBUG RGB - Blue band: min={blue_band.min():.4f}, max={blue_band.max():.4f}, mean={blue_band.mean():.4f}")

            # Normalizar RGB de forma simple y robusta
            # Sentinel-2 en DN tiene valores típicos entre 0-10000
            rgb_image = np.stack([red_band, green_band, blue_band], axis=2)

            print(f"      Band ranges - R:[{red_band.min():.0f},{red_band.max():.0f}] G:[{green_band.min():.0f},{green_band.max():.0f}] B:[{blue_band.min():.0f},{blue_band.max():.0f}]")

            # Usar percentiles para normalización robusta (evita valores extremos)
            p2, p98 = np.percentile(rgb_image, [2, 98])
            print(f"      RGB percentiles - P2:{p2:.0f}, P98:{p98:.0f}")

            # Normalizar a 0-1 usando percentiles
            rgb_normalized = (rgb_image - p2) / (p98 - p2 + 1e-10)  # +epsilon para evitar división por 0
            rgb_normalized = np.clip(rgb_normalized, 0, 1)

            # Ajuste gamma para mejorar contraste (gamma < 1 aclara, gamma > 1 oscurece)
            gamma = 0.8  # Aclara ligeramente la imagen
            rgb_normalized = np.power(rgb_normalized, gamma)

            # Convertir a uint8
            rgb_image = (rgb_normalized * 255).astype(np.uint8)

            print(f"      DEBUG RGB - Final image: min={rgb_image.min()}, max={rgb_image.max()}, mean={rgb_image.mean():.2f}")

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
            raise Exception(f"Error al obtener datos de Sentinel Hub: {str(e)}")

    def test_connection(self):
        """
        Probar la conexión con Sentinel Hub

        Returns:
            Dict con status de la conexión
        """
        try:
            # Intentar una request simple para verificar credenciales
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

            # Intentar obtener datos
            data = request.get_data()

            return {
                'status': 'success',
                'message': 'Conexión exitosa con Sentinel Hub',
                'data_shape': data[0].shape if data else None
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error de conexión: {str(e)}'
            }
