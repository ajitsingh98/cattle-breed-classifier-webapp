"""
FastAPI main application.
Cattle Breed Classifier API.
"""

import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.config import get_settings
from backend.app.core.logging import logger
from backend.app.api.routes import health, predict, metadata
from backend.app.services.inference import inference_service
from backend.app.services.breed_info import breed_info_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup
    logger.info("Starting Cattle Breed Classifier API...")
    try:
        inference_service.load()
        breed_info_service.load()
        logger.info("All services loaded successfully")
    except Exception as e:
        logger.error(f"Startup error: {e}")
        logger.info("API started with limited functionality")

    yield

    # Shutdown
    logger.info("Shutting down...")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "API for classifying Indian cattle breeds using deep learning. "
            "Supports image upload, URL, and base64 input. "
            "Returns breed prediction with confidence and metadata."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(health.router)
    app.include_router(predict.router)
    app.include_router(metadata.router)

    # Serve Static frontend if it exists
    static_dir = PROJECT_ROOT / "frontend" / "dist"
    if static_dir.exists() and static_dir.is_dir():
        from fastapi.staticfiles import StaticFiles
        from fastapi.responses import FileResponse
        
        # Mount assets
        assets_dir = static_dir / "assets"
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

        @app.get("/", tags=["Root"])
        async def root():
            return FileResponse(str(static_dir / "index.html"))
            
        # Catch-all for react router
        @app.get("/{full_path:path}", tags=["Root"])
        async def serve_spa(full_path: str):
            # Exclude api requests
            if full_path.startswith("api/") or full_path.startswith("docs") or full_path.startswith("redoc"):
                return {"detail": "Not Found"}
            return FileResponse(str(static_dir / "index.html"))
    else:
        @app.get("/", tags=["Root"])
        async def root():
            return {
                "message": "Cattle Breed Classifier API",
                "version": settings.app_version,
                "docs": "/docs",
            }

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "backend.app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
