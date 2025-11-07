"""
FastAPI main application
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import health, analysis
from config.config import Settings
from app.utils import setup_logging, get_logger

settings = Settings()
setup_logging(settings)
logger = get_logger(__name__)

from contextlib import asynccontextmanager

# Lifespan manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler - validates configuration on startup
    and logs shutdown
    """
    logger.info("Starting up FastAPI application")
    try:
        if not settings.validate_credentials():
            logger.warning("Sentinel Hub credentials not configured")
            logger.warning("Set SENTINEL_HUB_CLIENT_ID and SENTINEL_HUB_CLIENT_SECRET in .env")
            logger.warning("API will run but analysis endpoints will fail")
        else:
            logger.info("Configuration validated successfully")
            logger.info("CORS origins: %s", settings.cors_origins_list)
            logger.info("Analysis timeout: %d seconds", settings.analysis_timeout)
            logger.info("Sentinel Hub timeout: %d seconds", settings.sentinel_hub_timeout)
    except Exception as e:
        logger.error("Configuration error: %s", e, exc_info=True)
    
    yield
    
    logger.info("Shutting down FastAPI application")

app = FastAPI(
    title=settings.app_name,
    description=(
        "API para comparación de Region Growing Clásico vs Semántico "
        "en detección de estrés vegetal usando imágenes satelitales Sentinel-2"
    ),
    version=settings.app_version,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_tags=[
        {
            "name": "Health",
            "description": "Health check and connection test endpoints"
        },
        {
            "name": "Analysis",
            "description": "Vegetation stress analysis endpoints"
        }
    ],
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router)
app.include_router(analysis.router, prefix="/api/analysis")

# Root endpoint
@app.get("/", tags=["Health"])
async def root():
    """
    Root endpoint with API information
    """
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/api/docs",
        "health": "/health",
        "status": "running"
    }
