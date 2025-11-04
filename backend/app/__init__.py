from flask import Flask
from flask_cors import CORS
from flasgger import Swagger
from config.config import get_config


def create_app():
    """Factory para crear la aplicación Flask"""
    app = Flask(__name__)

    # Cargar configuración
    config = get_config()
    app.config.from_object(config)

    # Habilitar CORS
    CORS(app, origins=config.CORS_ORIGINS)

    # Configurar Swagger
    swagger_config = {
        "headers": [],
        "specs": [
            {
                "endpoint": 'apispec',
                "route": '/apispec.json',
                "rule_filter": lambda rule: True,
                "model_filter": lambda tag: True,
            }
        ],
        "static_url_path": "/flasgger_static",
        "swagger_ui": True,
        "specs_route": "/api/docs/"
    }

    swagger_template = {
        "swagger": "2.0",
        "info": {
            "title": "API de Detección de Estrés Vegetal",
            "description": "API para análisis de estrés vegetal usando Region Growing sobre imágenes satelitales Sentinel-2",
            "version": "1.0.0",
            "contact": {
                "name": "Proyecto de Maestría - Visión por Computadora"
            }
        },
        "schemes": ["http"],
        "tags": [
            {
                "name": "Health",
                "description": "Endpoints de verificación"
            },
            {
                "name": "Analysis",
                "description": "Endpoints de análisis de estrés vegetal"
            }
        ]
    }

    Swagger(app, config=swagger_config, template=swagger_template)

    # Registrar blueprints (controllers)
    from app.controllers import analysis_controller
    app.register_blueprint(analysis_controller.bp)

    # Ruta de prueba
    @app.route('/health', methods=['GET'])
    def health_check():
        """
        Health check endpoint
        ---
        tags:
          - Health
        responses:
          200:
            description: API funcionando correctamente
            schema:
              type: object
              properties:
                status:
                  type: string
                  example: "ok"
                message:
                  type: string
                  example: "API is running"
        """
        return {'status': 'ok', 'message': 'API is running'}, 200

    return app
