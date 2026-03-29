"""
Inference image preprocessing.
Handles loading images from file path, URL, or base64 string.
"""

import base64
from io import BytesIO
from pathlib import Path
from typing import Union

import requests
from PIL import Image
from torchvision import transforms

from ml.src.data.transforms import IMAGENET_MEAN, IMAGENET_STD


def load_image_from_path(path: str | Path) -> Image.Image:
    """Load an image from a file path."""
    return Image.open(path).convert('RGB')


def load_image_from_url(url: str, timeout: int = 10) -> Image.Image:
    """Load an image from a URL."""
    response = requests.get(url, timeout=timeout, stream=True)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert('RGB')


def load_image_from_base64(b64_string: str) -> Image.Image:
    """Load an image from a base64-encoded string."""
    # Remove data URI prefix if present
    if ',' in b64_string:
        b64_string = b64_string.split(',', 1)[1]
    image_bytes = base64.b64decode(b64_string)
    return Image.open(BytesIO(image_bytes)).convert('RGB')


def load_image_from_bytes(raw_bytes: bytes) -> Image.Image:
    """Load an image from raw bytes."""
    return Image.open(BytesIO(raw_bytes)).convert('RGB')


def get_inference_transform(img_size: int = 224) -> transforms.Compose:
    """Standard inference preprocessing pipeline."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def preprocess_image(
    image: Union[str, bytes, Image.Image],
    img_size: int = 224,
) -> tuple:
    """
    Universal image preprocessor.
    Accepts path string, URL, base64, bytes, or PIL Image.
    Returns (tensor, pil_image).
    """
    # Load image if needed
    if isinstance(image, str):
        if image.startswith(('http://', 'https://')):
            pil_image = load_image_from_url(image)
        elif image.startswith('data:image') or len(image) > 500:
            pil_image = load_image_from_base64(image)
        else:
            pil_image = load_image_from_path(image)
    elif isinstance(image, bytes):
        pil_image = load_image_from_bytes(image)
    elif isinstance(image, Image.Image):
        pil_image = image.convert('RGB')
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")

    transform = get_inference_transform(img_size)
    tensor = transform(pil_image).unsqueeze(0)  # Add batch dimension

    return tensor, pil_image
