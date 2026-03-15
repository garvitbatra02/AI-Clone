"""
Collection management routes for the Asset Upload Dashboard.

Prefix: /assets/collections
"""

from __future__ import annotations

import logging
from fastapi import APIRouter, HTTPException

from AssetUploadService.Server.models.schemas import (
    CollectionInfoSchema,
    CollectionListResponse,
    CreateCollectionRequest,
    CreateCollectionResponse,
    ErrorResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/assets/collections", tags=["Asset Collections"])


def _get_dashboard_service():
    """Lazy import to avoid circular imports at module level."""
    from AssetUploadService.services.dashboard_service import get_dashboard_service
    return get_dashboard_service()


# ==================== List Collections ====================


@router.get(
    "",
    response_model=CollectionListResponse,
    summary="List all collections",
    description=(
        "Returns every collection in the vector database along with its "
        "vector count, dimension, and distance metric."
    ),
)
async def list_collections() -> CollectionListResponse:
    """List all collections with stats."""
    try:
        service = _get_dashboard_service()
        names = await service.async_list_collections()
        
        items: list[CollectionInfoSchema] = []
        for name in names:
            stats = await service.async_get_collection_stats(name)
            items.append(
                CollectionInfoSchema(
                    collection=stats.get("collection", name),
                    exists=stats.get("exists", True),
                    vector_count=stats.get("vector_count", 0),
                    dimension=stats.get("dimension", 0),
                    distance_metric=stats.get("distance_metric", "cosine"),
                )
            )
        
        return CollectionListResponse(collections=items, total=len(items))
    
    except Exception as e:
        logger.error("Failed to list collections: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail={"error": str(e)})


# ==================== Get Single Collection ====================


@router.get(
    "/{collection_name}",
    response_model=CollectionInfoSchema,
    responses={404: {"model": ErrorResponse}},
    summary="Get collection stats",
    description="Get detailed statistics for a single collection.",
)
async def get_collection(collection_name: str) -> CollectionInfoSchema:
    """Get stats for one collection."""
    try:
        service = _get_dashboard_service()
        stats = await service.async_get_collection_stats(collection_name)
        
        if not stats.get("exists", False):
            raise HTTPException(
                status_code=404,
                detail={"error": f"Collection '{collection_name}' not found"},
            )
        
        return CollectionInfoSchema(
            collection=stats.get("collection", collection_name),
            exists=True,
            vector_count=stats.get("vector_count", 0),
            dimension=stats.get("dimension", 0),
            distance_metric=stats.get("distance_metric", "cosine"),
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get collection '%s': %s", collection_name, e, exc_info=True)
        raise HTTPException(status_code=500, detail={"error": str(e)})


# ==================== Create Collection ====================


@router.post(
    "",
    response_model=CreateCollectionResponse,
    status_code=201,
    summary="Create a collection",
    description="Explicitly create a new vector DB collection.",
)
async def create_collection(request: CreateCollectionRequest) -> CreateCollectionResponse:
    """Create a new collection."""
    try:
        service = _get_dashboard_service()
        result = await service.async_create_collection(request.name)
        
        return CreateCollectionResponse(
            created=result["created"],
            collection=result["collection"],
            reason=result.get("reason"),
        )
    
    except Exception as e:
        logger.error("Failed to create collection '%s': %s", request.name, e, exc_info=True)
        raise HTTPException(status_code=500, detail={"error": str(e)})


# ==================== Delete Collection ====================


@router.delete(
    "/{collection_name}",
    summary="Delete a collection",
    description="Permanently delete a collection and all its vectors.",
    responses={404: {"model": ErrorResponse}},
)
async def delete_collection(collection_name: str) -> dict:
    """Delete a collection."""
    try:
        service = _get_dashboard_service()
        deleted = await service.async_delete_collection(collection_name)
        
        if not deleted:
            raise HTTPException(
                status_code=404,
                detail={"error": f"Collection '{collection_name}' not found or already deleted"},
            )
        
        return {"deleted": True, "collection": collection_name}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete collection '%s': %s", collection_name, e, exc_info=True)
        raise HTTPException(status_code=500, detail={"error": str(e)})
