"""
Punto de entrada de la aplicación Flask
"""
from app import create_app
from config.config import get_config

# Crear instancia de la aplicación
app = create_app()
config = get_config()

if __name__ == '__main__':
    # Validar configuración antes de iniciar
    try:
        config.validate()
        print("[OK] All credentials configured")
    except ValueError as e:
        print(f"[WARNING] {e}")
        print("Tip: Configure Sentinel Hub credentials in .env for full functionality")
        print("You can still test the API endpoints\n")

    print(f"Starting Flask app in {config.FLASK_ENV} mode")
    print(f"API running on http://localhost:{config.PORT}")
    print(f"Health check: http://localhost:{config.PORT}/health")
    print(f"Test endpoint: http://localhost:{config.PORT}/api/analysis/test")
    print(f"Swagger UI: http://localhost:{config.PORT}/api/docs/")

    app.run(
        host='0.0.0.0',
        port=config.PORT,
        debug=config.DEBUG
    )
