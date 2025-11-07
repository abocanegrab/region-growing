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
