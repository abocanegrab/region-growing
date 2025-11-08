from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List


class Settings(BaseSettings):
    """
    Application settings using Pydantic Settings

    All settings are loaded from environment variables or .env file
    """
    model_config = SettingsConfigDict(
        env_file='backend/.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore'
    )

    # App
    app_name: str = "Sistema Híbrido de Detección de Estrés Vegetal"
    app_version: str = "2.0.0"
    debug: bool = False
    port: int = 8070

    # CORS - can be comma-separated string or list
    cors_origins: str = "http://localhost:3000,http://localhost:3001,http://localhost:3002,http://localhost:5173"

    # Sentinel Hub
    sentinel_hub_client_id: str = ""
    sentinel_hub_client_secret: str = ""

    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Timeouts (in seconds)
    sentinel_hub_timeout: int = 30
    analysis_timeout: int = 60
    
    @property
    def cors_origins_list(self) -> List[str]:
        """Convert CORS origins string to list"""
        if isinstance(self.cors_origins, str):
            return [origin.strip() for origin in self.cors_origins.split(',')]
        return self.cors_origins
    
    def validate_credentials(self) -> bool:
        """Check if Sentinel Hub credentials are configured"""
        return bool(self.sentinel_hub_client_id and self.sentinel_hub_client_secret)
