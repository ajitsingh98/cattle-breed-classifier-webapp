"""
Application configuration using Pydantic Settings.
Reads from environment variables and .env file.
"""

import os
from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration from environment variables."""

    # App
    app_name: str = "Cattle Breed Classifier API"
    app_version: str = "1.0.0"
    debug: bool = False

    # Model
    model_path: str = os.environ.get(
        "MODEL_PATH",
        str(Path(__file__).resolve().parents[3] / "models" / "cattle_breed_classifier_full_model.pth")
    )
    model_name: str = os.environ.get("MODEL_NAME", "resnet")
    model_version: str = os.environ.get("MODEL_VERSION", "v1.0")

    # Metadata
    metadata_path: str = os.environ.get(
        "METADATA_PATH",
        str(Path(__file__).resolve().parents[3] / "data" / "breed_metadata.csv")
    )
    classes_path: str = os.environ.get(
        "CLASSES_PATH",
        str(Path(__file__).resolve().parents[3] / "models" / "classes.txt")
    )

    # Image
    image_size: int = 224
    max_image_size_mb: float = 10.0

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = ["*"]

    # Inference
    top_k: int = 3
    url_timeout: int = 10

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
