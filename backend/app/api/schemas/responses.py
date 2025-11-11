"""
Pydantic response models for output validation
"""
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Health status", examples=["ok"])
    message: str = Field(..., description="Status message", examples=["Server is running"])


class Statistics(BaseModel):
    """Statistics from vegetation stress analysis"""
    
    total_area: float = Field(..., description="Total analyzed area in hectares")
    high_stress_area: float = Field(..., description="High stress area in hectares")
    medium_stress_area: float = Field(..., description="Medium stress area in hectares")
    low_stress_area: float = Field(..., description="Low stress area in hectares")
    mean_ndvi: float = Field(..., description="Mean NDVI value")
    num_regions: int = Field(..., description="Number of detected regions")
    cloud_coverage: float = Field(..., description="Cloud coverage percentage")
    date_from: str = Field(..., description="Analysis start date")
    date_to: str = Field(..., description="Analysis end date")


class AnalysisResponse(BaseModel):
    """Response model for analysis endpoint"""
    
    success: bool = Field(..., description="Whether analysis was successful")
    data: Optional[Dict[str, Any]] = Field(None, description="Analysis results")
    message: Optional[str] = Field(None, description="Status or error message")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "success": True,
                    "data": {
                        "geojson": {
                            "type": "FeatureCollection",
                            "features": []
                        },
                        "statistics": {
                            "total_area": 100.5,
                            "high_stress_area": 10.2,
                            "medium_stress_area": 30.5,
                            "low_stress_area": 59.8,
                            "mean_ndvi": 0.65,
                            "num_regions": 15,
                            "cloud_coverage": 5.2,
                            "date_from": "2024-01-01",
                            "date_to": "2024-01-31"
                        },
                        "images": {
                            "rgb": "base64_string",
                            "ndvi": "base64_string"
                        }
                    },
                    "message": None
                }
            ]
        }
    }


class ErrorResponse(BaseModel):
    """Error response model"""
    
    success: bool = Field(False, description="Always false for errors")
    error: str = Field(..., description="Error message")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "success": False,
                    "error": "Missing required field: bbox"
                }
            ]
        }
    }


class EmbeddingsMetadata(BaseModel):
    """Metadata for extracted embeddings"""
    
    bbox: Dict[str, float] = Field(..., description="Bounding box coordinates")
    date_from: str = Field(..., description="Start date")
    date_to: str = Field(..., description="End date")
    dimensions: tuple = Field(..., description="Image dimensions (width, height)")
    embeddings_shape: tuple = Field(..., description="Embeddings shape (H, W, D)")
    model: str = Field(..., description="Model used (simple or prithvi)")
    extracted_at: str = Field(..., description="Extraction timestamp (ISO format)")


class EmbeddingsExtractResponse(BaseModel):
    """Response model for embeddings extraction"""
    
    success: bool = Field(..., description="Whether extraction was successful")
    data: Optional[Dict[str, Any]] = Field(None, description="Extraction results")
    message: Optional[str] = Field(None, description="Status or error message")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "success": True,
                    "data": {
                        "embeddings_shape": [512, 512, 256],
                        "embeddings_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                        "download_url": "/api/embeddings/download/a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                        "metadata": {
                            "bbox": {
                                "min_lat": 32.45,
                                "min_lon": -115.35,
                                "max_lat": 32.55,
                                "max_lon": -115.25
                            },
                            "date_from": "2024-01-01",
                            "date_to": "2024-01-15",
                            "dimensions": [512, 512],
                            "embeddings_shape": [512, 512, 256],
                            "model": "prithvi",
                            "extracted_at": "2024-11-10T12:34:56.789"
                        }
                    },
                    "message": "Embeddings extracted successfully"
                }
            ]
        }
    }


class EmbeddingsSimilarityResponse(BaseModel):
    """Response model for similarity computation"""

    success: bool = Field(..., description="Whether computation was successful")
    data: Optional[Dict[str, Any]] = Field(None, description="Similarity results")
    message: Optional[str] = Field(None, description="Status or error message")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "success": True,
                    "data": {
                        "mean_similarity": 0.87,
                        "std_similarity": 0.12,
                        "min_similarity": 0.45,
                        "max_similarity": 0.99,
                        "similarity_map_shape": [512, 512],
                        "interpretation": "high_similarity",
                        "region_a_metadata": {},
                        "region_b_metadata": {}
                    },
                    "message": "Similarity computed successfully"
                }
            ]
        }
    }


class SegmentationMetricsSchema(BaseModel):
    """Schema for segmentation metrics"""

    num_regions: int = Field(..., description="Number of detected regions")
    coherence: float = Field(..., description="Spatial coherence percentage")
    avg_region_size: float = Field(..., description="Average region size in pixels")
    std_region_size: float = Field(..., description="Standard deviation of region sizes")
    largest_region_size: int = Field(..., description="Largest region size in pixels")
    smallest_region_size: int = Field(..., description="Smallest region size in pixels")
    processing_time: float = Field(..., description="Processing time in seconds")


class ComparisonMetrics(BaseModel):
    """Schema for comparison metrics between Classic RG and MGRG"""

    classic: SegmentationMetricsSchema = Field(..., description="Classic RG metrics")
    mgrg: SegmentationMetricsSchema = Field(..., description="MGRG metrics")
    differences: Dict[str, float] = Field(..., description="Differences between methods")
    winner: str = Field(..., description="Winner based on coherence (classic or mgrg)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "classic": {
                        "num_regions": 15,
                        "coherence": 72.5,
                        "avg_region_size": 680.3,
                        "std_region_size": 245.8,
                        "largest_region_size": 1200,
                        "smallest_region_size": 150,
                        "processing_time": 1.23
                    },
                    "mgrg": {
                        "num_regions": 3,
                        "coherence": 94.2,
                        "avg_region_size": 3400.5,
                        "std_region_size": 890.2,
                        "largest_region_size": 5000,
                        "smallest_region_size": 2100,
                        "processing_time": 1.45
                    },
                    "differences": {
                        "num_regions": -12,
                        "coherence": 21.7,
                        "avg_size": 2720.2,
                        "time": 0.22
                    },
                    "winner": "mgrg"
                }
            ]
        }
    }


class ComparisonResponse(BaseModel):
    """Response schema for comparison generation"""

    comparison_id: str = Field(..., description="Unique comparison identifier")
    status: str = Field(..., description="Processing status")
    message: str = Field(..., description="Status message")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "comparison_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                    "status": "processing",
                    "message": "Comparison started successfully"
                }
            ]
        }
    }
