from flask import Blueprint, request, jsonify
from flasgger import swag_from
from app.services.region_growing_service import RegionGrowingService

bp = Blueprint('analysis', __name__, url_prefix='/api/analysis')

# Inicializar servicio
region_growing_service = RegionGrowingService()


@bp.route('/analyze', methods=['POST'])
def analyze_region():
    """
    Analizar región del mapa para detectar estrés vegetal
    ---
    tags:
      - Analysis
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - bbox
          properties:
            bbox:
              type: object
              required:
                - min_lat
                - min_lon
                - max_lat
                - max_lon
              properties:
                min_lat:
                  type: number
                  description: Latitud mínima del bounding box
                  example: -12.0
                min_lon:
                  type: number
                  description: Longitud mínima del bounding box
                  example: -77.0
                max_lat:
                  type: number
                  description: Latitud máxima del bounding box
                  example: -11.9
                max_lon:
                  type: number
                  description: Longitud máxima del bounding box
                  example: -76.9
            date_from:
              type: string
              format: date
              description: Fecha inicial para búsqueda (YYYY-MM-DD)
              example: "2024-01-01"
            date_to:
              type: string
              format: date
              description: Fecha final para búsqueda (YYYY-MM-DD)
              example: "2024-01-31"
    responses:
      200:
        description: Análisis completado exitosamente
        schema:
          type: object
          properties:
            success:
              type: boolean
              example: true
            data:
              type: object
              properties:
                geojson:
                  type: object
                  description: GeoJSON con las regiones identificadas
                statistics:
                  type: object
                  description: Estadísticas del análisis
      400:
        description: Error en los parámetros de entrada
        schema:
          type: object
          properties:
            success:
              type: boolean
              example: false
            error:
              type: string
      500:
        description: Error interno del servidor
        schema:
          type: object
          properties:
            success:
              type: boolean
              example: false
            error:
              type: string
    """
    try:
        data = request.get_json()

        # Validar datos requeridos
        if not data or 'bbox' not in data:
            return jsonify({
                'success': False,
                'error': 'Se requiere el parámetro bbox con las coordenadas'
            }), 400

        bbox = data['bbox']
        required_bbox_fields = ['min_lat', 'min_lon', 'max_lat', 'max_lon']

        for field in required_bbox_fields:
            if field not in bbox:
                return jsonify({
                    'success': False,
                    'error': f'Falta el campo {field} en bbox'
                }), 400

        # Obtener fechas opcionales
        date_from = data.get('date_from')
        date_to = data.get('date_to')

        # Llamar al servicio
        result = region_growing_service.analyze_stress(
            bbox=bbox,
            date_from=date_from,
            date_to=date_to
        )

        return jsonify({
            'success': True,
            'data': result
        }), 200

    except ValueError as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error interno del servidor: {str(e)}'
        }), 500


@bp.route('/test', methods=['GET'])
def test():
    """
    Endpoint de prueba
    ---
    tags:
      - Analysis
    responses:
      200:
        description: Controller funcionando correctamente
        schema:
          type: object
          properties:
            success:
              type: boolean
              example: true
            message:
              type: string
              example: "Analysis controller is working"
    """
    return jsonify({
        'success': True,
        'message': 'Analysis controller is working'
    }), 200


@bp.route('/test-sentinel', methods=['GET'])
def test_sentinel():
    """
    Probar conexión con Sentinel Hub
    ---
    tags:
      - Analysis
    responses:
      200:
        description: Resultado de la prueba de conexión
        schema:
          type: object
          properties:
            success:
              type: boolean
            message:
              type: string
            data:
              type: object
    """
    try:
        result = region_growing_service.test_sentinel_connection()

        return jsonify({
            'success': result['status'] == 'success',
            'message': result['message'],
            'data': result
        }), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
