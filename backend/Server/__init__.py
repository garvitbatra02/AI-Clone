"""
FastAPI Server for ChatClone.

This package provides the REST API endpoints for the chat service.
"""

from .main import app, create_app

__all__ = ["app", "create_app"]
