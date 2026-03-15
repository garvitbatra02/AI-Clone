"""
Asset Upload Dashboard — Standalone FastAPI application.

Can be run independently for isolated development/testing::

    PYTHONPATH=. uvicorn AssetUploadService.Server.main:app --reload --port 8001

Or the routers can be imported into the main ChatClone app via
``get_asset_routers()`` (see ChatService/Server/main.py).
"""

from __future__ import annotations

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes.collections import router as collections_router
from .routes.uploads import router as uploads_router

logger = logging.getLogger(__name__)


def get_asset_routers():
    """
    Return the pair of asset-dashboard routers so the main app can
    mount them with ``app.include_router(r, prefix="/api")``.
    """
    return [collections_router, uploads_router]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise the dashboard service singleton at startup."""
    logger.info("Starting Asset Upload Dashboard service...")
    
    try:
        from AssetUploadService.services.dashboard_service import get_dashboard_service
        service = get_dashboard_service()
        
        supported = service.core.get_supported_file_types()
        logger.info("Asset Upload service ready — supported types: %s", supported)
        
        collections = await service.async_list_collections()
        if collections:
            logger.info("Existing collections: %s", collections)
        else:
            logger.info("No existing collections found")
    except Exception as e:
        logger.warning("Asset Upload service initialisation skipped: %s", e)
    
    yield
    
    logger.info("Shutting down Asset Upload Dashboard service...")


def create_asset_app() -> FastAPI:
    """
    Create a standalone FastAPI app for the Asset Upload Dashboard.
    
    This is useful when you want to run the asset dashboard on its own
    port without the Chat / RAG routes.
    """
    app = FastAPI(
        title="Asset Upload Dashboard API",
        description=(
            "## Asset Upload Dashboard\n\n"
            "Upload documents to your vector database, manage collections, "
            "and preview uploads before committing.\n\n"
            "### Features\n"
            "- **Collection management**: create, list, inspect, delete\n"
            "- **File uploads**: HTTP multipart & local-path\n"
            "- **Preview**: dry-run to see chunks before embedding\n"
            "- **Auto-detect**: file type detected from extension\n"
            "- **Text uploads**: raw text and batch\n"
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    # CORS
    origins = os.getenv(
        "CORS_ORIGINS", "http://localhost:3000,http://localhost:5173"
    ).split(",")
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Mount routers under /api so paths match the main app layout
    app.include_router(collections_router, prefix="/api")
    app.include_router(uploads_router, prefix="/api")
    
    @app.get("/health", tags=["Health"])
    async def health():
        return {"status": "healthy", "service": "asset-upload-dashboard"}
    
    return app


# App instance for ``uvicorn AssetUploadService.Server.main:app``
app = create_asset_app()
