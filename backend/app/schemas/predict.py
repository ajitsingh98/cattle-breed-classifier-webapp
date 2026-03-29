"""
Pydantic schemas for prediction endpoints.
"""

from typing import Optional
from pydantic import BaseModel, Field


class PredictionItem(BaseModel):
    """Single prediction result."""
    breed: str = Field(..., description="Predicted breed name")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")


class BreedInfo(BaseModel):
    """Breed metadata for the predicted breed."""
    breed_name: str
    animal_type: str
    region: str
    avg_milk_liters_per_day: str
    lifespan_years: str
    primary_use: str
    description: str


class PredictResponse(BaseModel):
    """Full prediction response."""
    predicted_breed: str
    confidence: float = Field(..., ge=0, le=1)
    top_k: list[PredictionItem]
    breed_info: Optional[BreedInfo] = None
    model_version: str = "v1.0"
    inference_time_ms: float
    warning: Optional[str] = None


class PredictURLRequest(BaseModel):
    """Request body for URL-based prediction."""
    url: str = Field(..., description="URL of the image to classify")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of top predictions")


class PredictBase64Request(BaseModel):
    """Request body for base64-based prediction."""
    image: str = Field(..., description="Base64-encoded image string")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of top predictions")


class ErrorResponse(BaseModel):
    """Error response."""
    detail: str
    error_type: str = "prediction_error"
