"""
Hierarchical analysis endpoint for end-to-end land cover and stress analysis.

This endpoint provides the complete analysis pipeline integrating:
- US-003: Sentinel-2 download
- US-006: Prithvi embeddings
- US-007: MGRG segmentation
- US-010: Semantic classification
- Stress analysis for crop regions
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Tuple, Dict, Any
import uuid
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from src.pipeline.hierarchical_analysis import (
    HierarchicalAnalysisPipeline,
    AnalysisConfig,
    AnalysisResult
)
from app.utils import get_logger

router = APIRouter()
logger = get_logger(__name__)


class HierarchicalAnalysisRequest(BaseModel):
    """
    Request schema for hierarchical analysis.

    Attributes
    ----------
    bbox : tuple
        Bounding box (min_lon, min_lat, max_lon, max_lat)
    date_from : str
        Start date YYYY-MM-DD
    date_to : str, optional
        End date (optional, defaults to date_from)
    mgrg_threshold : float
        MGRG similarity threshold (0.7-0.99)
    min_region_size : int
        Minimum region size in pixels
    export_formats : list
        Export formats: ["json", "tif", "png"]

    Examples
    --------
    >>> request = HierarchicalAnalysisRequest(
    ...     bbox=(-115.35, 32.45, -115.25, 32.55),
    ...     date_from="2025-10-15"
    ... )
    """
    bbox: Tuple[float, float, float, float] = Field(
        ...,
        description="Bounding box (min_lon, min_lat, max_lon, max_lat)",
        example=(-115.35, 32.45, -115.25, 32.55)
    )
    date_from: str = Field(
        ...,
        description="Start date YYYY-MM-DD",
        example="2025-10-15"
    )
    date_to: Optional[str] = Field(
        None,
        description="End date (optional, defaults to date_from)",
        example="2025-10-15"
    )
    mgrg_threshold: float = Field(
        0.95,
        ge=0.7,
        le=0.99,
        description="MGRG similarity threshold"
    )
    min_region_size: int = Field(
        50,
        ge=10,
        le=500,
        description="Minimum region size in pixels"
    )
    export_formats: List[str] = Field(
        ["json", "tif", "png"],
        description="Export formats"
    )

    @validator('bbox')
    def validate_bbox(cls, v):
        """Validate bbox coordinates."""
        min_lon, min_lat, max_lon, max_lat = v
        if not (-180 <= min_lon < max_lon <= 180):
            raise ValueError("Invalid longitude range")
        if not (-90 <= min_lat < max_lat <= 90):
            raise ValueError("Invalid latitude range")
        return v

    @validator('date_from', 'date_to')
    def validate_date(cls, v):
        """Validate date format."""
        if v is None:
            return v
        if not (len(v) == 10 and v[4] == '-' and v[7] == '-'):
            raise ValueError("Date must be YYYY-MM-DD format")
        return v


class HierarchicalAnalysisResponse(BaseModel):
    """
    Response schema for hierarchical analysis.

    Attributes
    ----------
    analysis_id : str
        Unique analysis identifier
    status : str
        Analysis status: "processing" | "completed" | "failed"
    message : str
        Status message
    output_files : dict, optional
        Output file paths
    summary : dict, optional
        Summary statistics
    """
    analysis_id: str
    status: str
    message: str
    output_files: Optional[Dict[str, str]] = None
    summary: Optional[Dict[str, float]] = None


# In-memory storage (replace with Redis/DB in production)
analysis_status: Dict[str, Dict[str, Any]] = {}


@router.post("/hierarchical", response_model=HierarchicalAnalysisResponse, tags=["Analysis"])
async def run_hierarchical_analysis(
    request: HierarchicalAnalysisRequest,
    background_tasks: BackgroundTasks
):
    """
    Run complete hierarchical analysis pipeline.

    This endpoint executes the complete analysis workflow:
    1. Download Sentinel-2 HLS imagery
    2. Extract Prithvi embeddings
    3. Segment with MGRG
    4. Classify objects semantically
    5. Analyze stress levels (crops only)
    6. Generate outputs (JSON, GeoTIFF, PNG)

    The analysis runs asynchronously in the background. Use the returned
    analysis_id to check status and download results.

    Parameters
    ----------
    request : HierarchicalAnalysisRequest
        Analysis request parameters
    background_tasks : BackgroundTasks
        FastAPI background tasks

    Returns
    -------
    HierarchicalAnalysisResponse
        Response with analysis_id and status

    Raises
    ------
    HTTPException
        400 for validation errors
        500 for processing errors

    Examples
    --------
    >>> response = await run_hierarchical_analysis(request)
    >>> print(response.analysis_id)
    '123e4567-e89b-12d3-a456-426614174000'
    >>> print(response.status)
    'processing'
    """
    # Generate unique ID
    analysis_id = str(uuid.uuid4())

    logger.info(f"Starting hierarchical analysis: {analysis_id}")
    logger.info(f"BBox: {request.bbox}")
    logger.info(f"Date: {request.date_from}")

    try:
        # Create config
        config = AnalysisConfig(
            bbox=request.bbox,
            date_from=request.date_from,
            date_to=request.date_to,
            mgrg_threshold=request.mgrg_threshold,
            min_region_size=request.min_region_size,
            export_formats=request.export_formats,
            output_dir=f"output/api/{analysis_id}"
        )

        # Initialize status
        analysis_status[analysis_id] = {
            'status': 'processing',
            'message': 'Analysis started',
        }

        # Run pipeline in background
        background_tasks.add_task(
            run_pipeline_background,
            analysis_id,
            config
        )

        return HierarchicalAnalysisResponse(
            analysis_id=analysis_id,
            status="processing",
            message="Analysis started. Use GET /api/analysis/hierarchical/{id}/status to check progress."
        )

    except Exception as e:
        logger.error(f"Failed to start analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def run_pipeline_background(analysis_id: str, config: AnalysisConfig):
    """
    Run pipeline in background task.

    Parameters
    ----------
    analysis_id : str
        Unique analysis identifier
    config : AnalysisConfig
        Analysis configuration
    """
    try:
        logger.info(f"Running pipeline for analysis: {analysis_id}")

        pipeline = HierarchicalAnalysisPipeline(config)
        result = pipeline.run()

        # Update status
        analysis_status[analysis_id] = {
            'status': 'completed',
            'message': 'Analysis completed successfully',
            'output_files': result.output_files,
            'summary': result.summary,
            'processing_time': result.processing_time,
        }

        logger.info(f"Analysis completed: {analysis_id}")

    except Exception as e:
        logger.error(f"Analysis failed: {analysis_id}: {e}", exc_info=True)
        analysis_status[analysis_id] = {
            'status': 'failed',
            'message': f"Analysis failed: {str(e)}",
        }


@router.get("/hierarchical/{analysis_id}/status", tags=["Analysis"])
async def get_analysis_status(analysis_id: str):
    """
    Get status of analysis.

    Parameters
    ----------
    analysis_id : str
        Analysis identifier

    Returns
    -------
    dict
        Status information

    Raises
    ------
    HTTPException
        404 if analysis not found

    Examples
    --------
    >>> status = await get_analysis_status("123e4567-e89b-12d3-a456-426614174000")
    >>> print(status['status'])
    'completed'
    """
    if analysis_id not in analysis_status:
        logger.warning(f"Analysis not found: {analysis_id}")
        raise HTTPException(status_code=404, detail="Analysis not found")

    return analysis_status[analysis_id]


@router.get("/hierarchical/{analysis_id}/download/{file_type}", tags=["Analysis"])
async def download_analysis_result(analysis_id: str, file_type: str):
    """
    Download result file.

    Parameters
    ----------
    analysis_id : str
        Analysis identifier
    file_type : str
        File type: "json", "tif", "png"

    Returns
    -------
    FileResponse
        File download response

    Raises
    ------
    HTTPException
        404 if analysis or file not found
        400 if analysis not completed

    Examples
    --------
    >>> response = await download_analysis_result(
    ...     "123e4567-e89b-12d3-a456-426614174000",
    ...     "json"
    ... )
    """
    if analysis_id not in analysis_status:
        raise HTTPException(status_code=404, detail="Analysis not found")

    status = analysis_status[analysis_id]
    if status['status'] != 'completed':
        raise HTTPException(
            status_code=400,
            detail=f"Analysis not completed. Status: {status['status']}"
        )

    # Get file path
    output_files = status.get('output_files', {})
    file_path = output_files.get(file_type)

    if not file_path or not Path(file_path).exists():
        raise HTTPException(
            status_code=404,
            detail=f"File type {file_type} not found"
        )

    # Determine media type
    media_types = {
        'json': 'application/json',
        'tif': 'image/tiff',
        'png': 'image/png',
        'html': 'text/html',
    }

    logger.info(f"Downloading file: {file_path}")

    return FileResponse(
        file_path,
        media_type=media_types.get(file_type, 'application/octet-stream'),
        filename=Path(file_path).name
    )
