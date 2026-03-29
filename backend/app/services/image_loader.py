"""
Image loading service.
Handles loading images from file upload, URL, and base64.
"""

import base64
from io import BytesIO

import requests
from PIL import Image

from backend.app.core.config import get_settings
from backend.app.core.logging import logger


def load_image_from_upload(file_bytes: bytes) -> Image.Image:
    """Load image from uploaded file bytes."""
    try:
        image = Image.open(BytesIO(file_bytes)).convert('RGB')
        return image
    except Exception as e:
        logger.error(f"Failed to load uploaded image: {e}")
        raise ValueError(f"Invalid image file: {e}")


def load_image_from_url(url: str) -> Image.Image:
    """Load image from a URL."""
    settings = get_settings()
    try:
        response = requests.get(url, timeout=settings.url_timeout, stream=True)
        response.raise_for_status()

        # Check content length
        content_length = response.headers.get('Content-Length')
        if content_length and int(content_length) > settings.max_image_size_mb * 1024 * 1024:
            raise ValueError(f"Image too large (max {settings.max_image_size_mb}MB)")

        image = Image.open(BytesIO(response.content)).convert('RGB')
        return image
    except requests.RequestException as e:
        logger.error(f"Failed to fetch image from URL: {e}")
        raise ValueError(f"Could not fetch image from URL: {e}")
    except Exception as e:
        logger.error(f"Failed to load image from URL: {e}")
        raise ValueError(f"Invalid image from URL: {e}")


def load_image_from_base64(b64_string: str) -> Image.Image:
    """Load image from base64-encoded string."""
    try:
        # Remove data URI prefix if present
        if ',' in b64_string:
            b64_string = b64_string.split(',', 1)[1]

        image_bytes = base64.b64decode(b64_string)
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        return image
    except Exception as e:
        logger.error(f"Failed to load base64 image: {e}")
        raise ValueError(f"Invalid base64 image: {e}")
