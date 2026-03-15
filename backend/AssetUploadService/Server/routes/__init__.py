"""Asset Upload Service — API routes."""

from .collections import router as collections_router
from .uploads import router as uploads_router

__all__ = ["collections_router", "uploads_router"]
