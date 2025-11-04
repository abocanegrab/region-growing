import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()


class Config:
    """Configuración base de la aplicación"""

    # Flask
    FLASK_ENV = os.getenv('FLASK_ENV', 'development')
    DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    PORT = int(os.getenv('FLASK_PORT', 5000))

    # Sentinel Hub
    SENTINEL_HUB_CLIENT_ID = os.getenv('SENTINEL_HUB_CLIENT_ID')
    SENTINEL_HUB_CLIENT_SECRET = os.getenv('SENTINEL_HUB_CLIENT_SECRET')

    # CORS
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', 'http://localhost:5173').split(',')

    # Database (para futuro)
    # DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///database.db')

    @staticmethod
    def validate():
        """Valida que las configuraciones necesarias estén presentes"""
        required = [
            'SENTINEL_HUB_CLIENT_ID',
            'SENTINEL_HUB_CLIENT_SECRET'
        ]

        missing = []
        for key in required:
            if not os.getenv(key):
                missing.append(key)

        if missing:
            raise ValueError(
                f"Faltan las siguientes variables de entorno: {', '.join(missing)}\n"
                f"Por favor, configura el archivo .env basándote en .env.example"
            )


class DevelopmentConfig(Config):
    """Configuración para desarrollo"""
    DEBUG = True


class ProductionConfig(Config):
    """Configuración para producción"""
    DEBUG = False


# Mapeo de configuraciones
config_by_name = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}


def get_config():
    """Retorna la configuración según el ambiente"""
    env = os.getenv('FLASK_ENV', 'development')
    return config_by_name.get(env, DevelopmentConfig)
