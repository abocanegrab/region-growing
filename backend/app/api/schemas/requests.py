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
