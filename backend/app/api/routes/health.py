"""
Health and version endpoints.
"""

from fastapi import APIRouter

from backend.app.core.config import get_settings
from backend.app.services.inference import inference_service

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": inference_service.is_loaded,
    }


@router.get("/version")
async def version():
    """API version info."""
    settings = get_settings()
    return {
        "app_name": settings.app_name,
        "app_version": settings.app_version,
        "model_version": settings.model_version,
        "model_name": settings.model_name,
        "num_classes": len(inference_service.class_names),
    }
