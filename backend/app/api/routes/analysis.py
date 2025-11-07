"""
Analysis endpoints for vegetation stress detection
"""
import asyncio
from fastapi import APIRouter, HTTPException
from app.api.schemas.requests import AnalysisRequest
from app.api.schemas.responses import AnalysisResponse
from app.services.region_growing_service import RegionGrowingService
from app.utils import get_logger, TimeoutError
from config.config import Settings

router = APIRouter()
logger = get_logger(__name__)
settings = Settings()


@router.post("/analyze", response_model=AnalysisResponse, tags=["Analysis"])
async def analyze_region(request: AnalysisRequest):
    """
    Analyze vegetation stress in a geographic region using Region Growing algorithm

    This endpoint:
    1. Downloads Sentinel-2 satellite imagery for the specified bounding box
    2. Calculates NDVI (Normalized Difference Vegetation Index)
    3. Applies Region Growing algorithm to segment vegetation areas
    4. Classifies regions by stress level (high/medium/low)
    5. Returns GeoJSON with regions and statistics

    The analysis has a timeout configured in settings (default 60 seconds).
    If the analysis exceeds this timeout, a 504 error will be returned.

    Args:
        request: Analysis request with bbox coordinates and optional date range

    Returns:
        AnalysisResponse: Analysis results with GeoJSON, statistics, and images

    Raises:
        HTTPException: 400 for validation errors, 504 for timeout, 500 for processing errors
    """
    logger.info("Starting analysis for bbox: %s", request.bbox)

    try:
        service = RegionGrowingService()

        # Convert bbox to dict format expected by service
        bbox_dict = {
            'min_lat': request.bbox.min_lat,
            'min_lon': request.bbox.min_lon,
            'max_lat': request.bbox.max_lat,
            'max_lon': request.bbox.max_lon
        }

        # Run analysis with timeout
        try:
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None,
                    service.analyze_stress,
                    bbox_dict,
                    request.date_from,
                    request.date_to
                ),
                timeout=settings.analysis_timeout
            )

            logger.info("Analysis completed successfully")
            return AnalysisResponse(
                success=True,
                data=result,
                message=None
            )

        except asyncio.TimeoutError:
            error_msg = (
                f"Analysis timed out after {settings.analysis_timeout} seconds. "
                f"The requested area may be too large or the date range too wide. "
                f"Try reducing the area or selecting a narrower date range."
            )
            logger.error("Analysis timeout: %s", error_msg)
            raise HTTPException(status_code=504, detail=error_msg)

    except ValueError as e:
        logger.warning("Validation error: %s", str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error("Internal server error: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
