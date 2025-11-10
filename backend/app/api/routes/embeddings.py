"""
Embeddings API endpoints for HLS image processing.
"""
import logging
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import FileResponse
import os

from backend.app.api.schemas.requests import (
    EmbeddingsExtractRequest,
    EmbeddingsSimilarityRequest
)
from backend.app.api.schemas.responses import (
    EmbeddingsExtractResponse,
    EmbeddingsSimilarityResponse,
    ErrorResponse
)
from backend.app.services.embeddings_service import EmbeddingsService

logger = logging.getLogger(__name__)

router = APIRouter()


def get_embeddings_service() -> EmbeddingsService:
    """
    Dependency to get embeddings service instance.
    """
    client_id = os.getenv('SENTINELHUB_CLIENT_ID')
    client_secret = os.getenv('SENTINELHUB_CLIENT_SECRET')
    
    if not client_id or not client_secret:
        raise HTTPException(
            status_code=500,
            detail="Sentinel Hub credentials not configured"
        )
    
    use_simple_model = os.getenv('USE_SIMPLE_MODEL', 'false').lower() == 'true'
    
    return EmbeddingsService(
        sentinel_client_id=client_id,
        sentinel_client_secret=client_secret,
        use_simple_model=use_simple_model
    )


@router.post(
    "/extract",
    response_model=EmbeddingsExtractResponse,
    responses={
        200: {"description": "Embeddings extracted successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="Extract embeddings from HLS image",
    description="""
    Extract semantic embeddings from a geographic region using Prithvi model.
    
    This endpoint:
    1. Downloads HLS bands from Sentinel Hub for the specified bbox
    2. Prepares HLS image (resamples 20m bands to 10m)
    3. Extracts 256-dimensional embeddings using Prithvi foundation model
    4. Saves embeddings to disk and returns metadata with download URL
    
    The embeddings can be used for:
    - Semantic segmentation (MGRG algorithm)
    - Region similarity comparison
    - Object detection in agricultural scenes
    """
)
async def extract_embeddings(
    request: EmbeddingsExtractRequest,
    service: EmbeddingsService = Depends(get_embeddings_service)
):
    """
    Extract embeddings from HLS image.
    """
    try:
        logger.info(f"Extracting embeddings for bbox: {request.bbox.model_dump()}")
        
        service.use_simple_model = request.use_simple_model
        
        embeddings, metadata = service.extract_embeddings_from_bbox(
            bbox_coords=request.bbox.model_dump(),
            date_from=request.date_from,
            date_to=request.date_to
        )
        
        embedding_id, file_path = service.save_embeddings_to_disk(
            embeddings,
            metadata
        )
        
        return EmbeddingsExtractResponse(
            success=True,
            data={
                "embeddings_shape": list(embeddings.shape),
                "embeddings_id": embedding_id,
                "download_url": f"/api/embeddings/download/{embedding_id}",
                "metadata": metadata
            },
            message="Embeddings extracted successfully"
        )
        
    except Exception as e:
        logger.error(f"Failed to extract embeddings: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/download/{embedding_id}",
    response_class=FileResponse,
    responses={
        200: {"description": "Embeddings file (.npz)"},
        404: {"model": ErrorResponse, "description": "Embeddings not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="Download embeddings file",
    description="""
    Download embeddings as a compressed numpy file (.npz).
    
    The file contains:
    - 'embeddings': numpy array with shape (H, W, 256)
    - 'metadata_*': various metadata fields
    
    Load in Python with:
    ```python
    import numpy as np
    data = np.load('embeddings.npz')
    embeddings = data['embeddings']
    metadata = {k.replace('metadata_', ''): data[k] for k in data.files if k.startswith('metadata_')}
    ```
    """
)
async def download_embeddings(
    embedding_id: str,
    service: EmbeddingsService = Depends(get_embeddings_service)
):
    """
    Download embeddings file by ID.
    """
    try:
        file_path = service.get_embeddings_file_path(embedding_id)
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Embeddings not found")
        
        return FileResponse(
            path=str(file_path),
            filename=f"embeddings_{embedding_id}.npz",
            media_type="application/octet-stream"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/similarity",
    response_model=EmbeddingsSimilarityResponse,
    responses={
        200: {"description": "Similarity computed successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="Compute similarity between two regions",
    description="""
    Compute cosine similarity between embeddings of two geographic regions.
    
    This endpoint:
    1. Extracts embeddings for both regions
    2. Crops to same dimensions
    3. Computes pixelwise cosine similarity
    4. Returns statistical summary
    
    Similarity interpretation:
    - > 0.9: Very high similarity (same land cover type)
    - 0.7-0.9: High similarity (similar vegetation/crops)
    - 0.5-0.7: Medium similarity (different but related)
    - < 0.5: Low similarity (different land cover types)
    """
)
async def compute_similarity(
    request: EmbeddingsSimilarityRequest,
    service: EmbeddingsService = Depends(get_embeddings_service)
):
    """
    Compute similarity between two regions.
    """
    try:
        logger.info(f"Computing similarity between regions")
        logger.info(f"  Region A: {request.bbox_a.model_dump()}")
        logger.info(f"  Region B: {request.bbox_b.model_dump()}")
        
        results = service.compute_similarity_between_regions(
            bbox_a=request.bbox_a.model_dump(),
            bbox_b=request.bbox_b.model_dump(),
            date_from=request.date_from,
            date_to=request.date_to
        )
        
        mean_sim = results['mean_similarity']
        
        if mean_sim > 0.9:
            interpretation = "very_high_similarity"
        elif mean_sim > 0.7:
            interpretation = "high_similarity"
        elif mean_sim > 0.5:
            interpretation = "medium_similarity"
        else:
            interpretation = "low_similarity"
        
        results['interpretation'] = interpretation
        
        return EmbeddingsSimilarityResponse(
            success=True,
            data=results,
            message="Similarity computed successfully"
        )
        
    except Exception as e:
        logger.error(f"Failed to compute similarity: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
