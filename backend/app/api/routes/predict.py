"""
Prediction endpoints.
POST /predict/file  - Upload image file
POST /predict/url   - Predict from image URL
POST /predict/base64 - Predict from base64 image
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Query

from backend.app.schemas.predict import (
    PredictResponse,
    PredictURLRequest,
    PredictBase64Request,
    PredictionItem,
    BreedInfo,
)
from backend.app.services.inference import inference_service
from backend.app.services.image_loader import (
    load_image_from_upload,
    load_image_from_url,
    load_image_from_base64,
)
from backend.app.services.breed_info import breed_info_service
from backend.app.core.logging import logger

router = APIRouter(prefix="/predict", tags=["Prediction"])


def _build_response(result: dict) -> PredictResponse:
    """Build a PredictResponse from inference result dict."""
    # Get breed info
    breed_info = None
    breed_summary = breed_info_service.get_breed_summary(result['predicted_breed'])
    if breed_summary:
        breed_info = BreedInfo(**breed_summary)

    return PredictResponse(
        predicted_breed=result['predicted_breed'],
        confidence=result['confidence'],
        top_k=[PredictionItem(**item) for item in result['top_k']],
        breed_info=breed_info,
        model_version=result.get('model_version', 'v1.0'),
        inference_time_ms=result['inference_time_ms'],
        warning=result.get('warning'),
    )


@router.post("/file", response_model=PredictResponse)
async def predict_file(
    file: UploadFile = File(...),
    top_k: int = Query(default=3, ge=1, le=10),
):
    """Predict breed from uploaded image file."""
    if not inference_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        contents = await file.read()
        image = load_image_from_upload(contents)
        result = inference_service.predict(image, top_k=top_k)
        return _build_response(result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")


@router.post("/url", response_model=PredictResponse)
async def predict_url(request: PredictURLRequest):
    """Predict breed from image URL."""
    if not inference_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        image = load_image_from_url(request.url)
        result = inference_service.predict(image, top_k=request.top_k)
        return _build_response(result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")


@router.post("/base64", response_model=PredictResponse)
async def predict_base64(request: PredictBase64Request):
    """Predict breed from base64-encoded image."""
    if not inference_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        image = load_image_from_base64(request.image)
        result = inference_service.predict(image, top_k=request.top_k)
        return _build_response(result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")
