"""
Pydantic request models for input validation
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional


class BBoxRequest(BaseModel):
    """Bounding box coordinates for geographic region"""
    
    min_lat: float = Field(..., ge=-90, le=90, description="Minimum latitude")
    min_lon: float = Field(..., ge=-180, le=180, description="Minimum longitude")
    max_lat: float = Field(..., ge=-90, le=90, description="Maximum latitude")
    max_lon: float = Field(..., ge=-180, le=180, description="Maximum longitude")
    
    @field_validator('max_lat')
    @classmethod
    def validate_lat_order(cls, v: float, info) -> float:
        """Validate that max_lat is greater than min_lat"""
        if 'min_lat' in info.data and v <= info.data['min_lat']:
            raise ValueError('max_lat must be greater than min_lat')
        return v
    
    @field_validator('max_lon')
    @classmethod
    def validate_lon_order(cls, v: float, info) -> float:
        """Validate that max_lon is greater than min_lon"""
        if 'min_lon' in info.data and v <= info.data['min_lon']:
            raise ValueError('max_lon must be greater than min_lon')
        return v


class AnalysisRequest(BaseModel):
    """Request model for vegetation stress analysis"""
    
    bbox: BBoxRequest = Field(..., description="Bounding box coordinates")
    date_from: Optional[str] = Field(
        None, 
        description="Start date for image search (YYYY-MM-DD)",
        pattern=r'^\d{4}-\d{2}-\d{2}$'
    )
    date_to: Optional[str] = Field(
        None, 
        description="End date for image search (YYYY-MM-DD)",
        pattern=r'^\d{4}-\d{2}-\d{2}$'
    )
    method: str = Field(
        "classic", 
        description="Analysis method: 'classic' or 'hybrid'",
        pattern=r'^(classic|hybrid)$'
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "bbox": {
                        "min_lat": -12.05,
                        "min_lon": -77.05,
                        "max_lat": -11.95,
                        "max_lon": -76.95
                    },
                    "date_from": "2024-01-01",
                    "date_to": "2024-01-31",
                    "method": "classic"
                }
            ]
        }
    }


class EmbeddingsExtractRequest(BaseModel):
    """Request model for embeddings extraction"""
    
    bbox: BBoxRequest = Field(..., description="Bounding box coordinates")
    date_from: Optional[str] = Field(
        None,
        description="Start date for image search (YYYY-MM-DD)",
        pattern=r'^\d{4}-\d{2}-\d{2}$'
    )
    date_to: Optional[str] = Field(
        None,
        description="End date for image search (YYYY-MM-DD)",
        pattern=r'^\d{4}-\d{2}-\d{2}$'
    )
    use_simple_model: bool = Field(
        False,
        description="Use simplified model for testing (no Prithvi download needed)"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "bbox": {
                        "min_lat": 32.45,
                        "min_lon": -115.35,
                        "max_lat": 32.55,
                        "max_lon": -115.25
                    },
                    "date_from": "2024-01-01",
                    "date_to": "2024-01-31",
                    "use_simple_model": False
                }
            ]
        }
    }


class EmbeddingsSimilarityRequest(BaseModel):
    """Request model for computing similarity between two regions"""

    bbox_a: BBoxRequest = Field(..., description="First bounding box")
    bbox_b: BBoxRequest = Field(..., description="Second bounding box")
    date_from: Optional[str] = Field(
        None,
        description="Start date for image search (YYYY-MM-DD)",
        pattern=r'^\d{4}-\d{2}-\d{2}$'
    )
    date_to: Optional[str] = Field(
        None,
        description="End date for image search (YYYY-MM-DD)",
        pattern=r'^\d{4}-\d{2}-\d{2}$'
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "bbox_a": {
                        "min_lat": 32.45,
                        "min_lon": -115.35,
                        "max_lat": 32.55,
                        "max_lon": -115.25
                    },
                    "bbox_b": {
                        "min_lat": 20.85,
                        "min_lon": -101.45,
                        "max_lat": 20.95,
                        "max_lon": -101.35
                    },
                    "date_from": "2024-01-01",
                    "date_to": "2024-01-31"
                }
            ]
        }
    }


class ComparisonRequest(BaseModel):
    """Request schema for A/B comparison between Classic RG and MGRG"""

    bbox: BBoxRequest = Field(..., description="Bounding box coordinates")
    date_from: str = Field(
        ...,
        description="Start date in YYYY-MM-DD format",
        pattern=r'^\d{4}-\d{2}-\d{2}$'
    )
    date_to: Optional[str] = Field(
        None,
        description="End date in YYYY-MM-DD format (defaults to date_from)",
        pattern=r'^\d{4}-\d{2}-\d{2}$'
    )
    classic_threshold: float = Field(
        0.1,
        description="NDVI threshold for classic RG",
        ge=0.0,
        le=1.0
    )
    mgrg_threshold: float = Field(
        0.85,
        description="Cosine similarity threshold for MGRG",
        ge=0.0,
        le=1.0
    )
    seed_method: str = Field(
        "grid",
        description="Seed generation method (grid or kmeans)",
        pattern=r'^(grid|kmeans)$'
    )
    export_formats: list = Field(
        ["png"],
        description="Export formats (png, pdf, svg)"
    )
    dpi: int = Field(
        300,
        description="Resolution for raster exports",
        ge=72,
        le=600
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "bbox": {
                        "min_lat": 32.45,
                        "min_lon": -115.35,
                        "max_lat": 32.55,
                        "max_lon": -115.25
                    },
                    "date_from": "2024-01-15",
                    "date_to": "2024-01-15",
                    "classic_threshold": 0.1,
                    "mgrg_threshold": 0.85,
                    "seed_method": "kmeans",
                    "export_formats": ["png", "pdf"],
                    "dpi": 300
                }
            ]
        }
    }
