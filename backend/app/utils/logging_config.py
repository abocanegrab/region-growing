"""
Logging configuration module for the application
"""
import logging
import sys
from typing import Optional
from config.config import Settings


def setup_logging(settings: Optional[Settings] = None) -> None:
    """
    Configure logging for the entire application

    Args:
        settings: Application settings instance (optional)
    """
    if settings is None:
        settings = Settings()

    # Get log level from settings
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format=settings.log_format,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Set specific log levels for external libraries to reduce noise
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module

    Args:
        name: Name of the module (usually __name__)

    Returns:
        Logger instance configured with application settings
    """
    return logging.getLogger(name)
