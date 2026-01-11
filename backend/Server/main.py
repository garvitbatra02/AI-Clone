"""
FastAPI Server for ChatClone.

This is the main entry point for the API server.
Provides endpoints for chat inference with automatic provider rotation.
"""

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from ChatServer import get_chat_service
from .routes.chat import router as chat_router
from .models.schemas import HealthResponse, ServiceInfoResponse

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Initializes services on startup and cleans up on shutdown.
    """
    # Startup
    logger.info("Starting ChatClone API Server...")
    
    # Initialize the chat service (singleton)
    service = get_chat_service()
    logger.info(f"Chat service initialized with {service.get_provider_count()} providers")
    logger.info(f"Active providers: {[p.value for p in service.get_active_providers()]}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down ChatClone API Server...")
    service.clear_cache()


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI application instance.
    """
    app = FastAPI(
        title="ChatClone API",
        description="""
        ## Chat API with LLM Provider Rotation
        
        This API provides chat inference endpoints that automatically rotate
        between multiple LLM providers (Groq, Cerebras) to manage rate limits.
        
        ### Features:
        - **Automatic Provider Rotation**: Distributes requests across providers
        - **Fallback Support**: Tries next provider on failure
        - **Streaming**: Real-time response streaming via SSE or WebSocket
        - **Async Support**: High-performance async endpoints
        
        ### Providers:
        - **Groq**: Fast inference with Llama models
        - **Cerebras**: Ultra-fast inference with Llama models
        """,
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    # Configure CORS
    origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",")
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(chat_router, prefix="/api")
    
    # Health check endpoint
    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["Health"],
        summary="Health check",
        description="Check if the service is healthy and list available providers.",
    )
    async def health_check() -> HealthResponse:
        """Health check endpoint."""
        service = get_chat_service()
        return HealthResponse(
            status="healthy",
            providers=[p.value for p in service.get_active_providers()],
            provider_count=service.get_provider_count(),
        )
    
    # Service info endpoint
    @app.get(
        "/",
        response_model=ServiceInfoResponse,
        tags=["Info"],
        summary="Service information",
        description="Get information about the service and available providers.",
    )
    async def service_info() -> ServiceInfoResponse:
        """Get service information."""
        service = get_chat_service()
        return ServiceInfoResponse(
            name="ChatClone API",
            version="1.0.0",
            providers=[p.value for p in service.get_active_providers()],
            current_provider=service.get_current_provider().provider.value,
        )
    
    return app


# Create the app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "true").lower() == "true"
    
    logger.info(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        "Server.main:app",
        host=host,
        port=port,
        reload=reload,
    )
