"""
Health check endpoints
"""
from fastapi import APIRouter
from app.api.schemas.responses import HealthResponse
from app.services.region_growing_service import RegionGrowingService

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint to verify the API is running
    
    Returns:
        HealthResponse: Server status
    """
    return HealthResponse(
        status="ok",
        message="Server is running"
    )


@router.get("/api/analysis/test", tags=["Health"])
async def test_sentinel_hub():
    """
    Test Sentinel Hub API connection
    
    Returns:
        dict: Connection test result with status and message
    """
    service = RegionGrowingService()
    result = service.test_sentinel_connection()
    
    return {
        "success": result['status'] == 'success',
        "message": result['message'],
        "data": result
    }
