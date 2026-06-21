"""
Collection management routes for the Asset Upload Dashboard.

Prefix: /assets/collections
"""

from __future__ import annotations

import logging
from fastapi import APIRouter, HTTPException, Query

from AssetUploadService.Server.models.schemas import (
    CollectionInfoSchema,
    CollectionListResponse,
    CreateCollectionRequest,
    CreateCollectionResponse,
    DeleteFileResponse,
    ErrorResponse,
    FileChunkSchema,
    FileChunksResponse,
    FileInfoSchema,
    FileListResponse,
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


# ==================== List Files in a Collection ====================


@router.get(
    "/{collection_name}/files",
    response_model=FileListResponse,
    responses={404: {"model": ErrorResponse}},
    summary="List files in a collection",
    description=(
        "Returns the distinct source files inside a collection by grouping "
        "the stored vector points on their 'source' payload field. Each entry "
        "includes the chunk count and (when available) file type, upload time, "
        "and size."
    ),
)
async def list_files(collection_name: str) -> FileListResponse:
    """List the distinct files stored in a collection."""
    try:
        service = _get_dashboard_service()
        stats = await service.async_get_collection_stats(collection_name)
        if not stats.get("exists", False):
            raise HTTPException(
                status_code=404,
                detail={"error": f"Collection '{collection_name}' not found"},
            )
        
        files = await service.async_list_files(collection_name)
        items = [FileInfoSchema(**f) for f in files]
        return FileListResponse(files=items, total=len(items))
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to list files in '%s': %s", collection_name, e, exc_info=True)
        raise HTTPException(status_code=500, detail={"error": str(e)})


# ==================== Get a File's Chunks ====================


@router.get(
    "/{collection_name}/files/chunks",
    response_model=FileChunksResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Get the chunks of a single file",
    description=(
        "Returns every chunk belonging to one source file, ordered by "
        "chunk_index. Used to power the Asset Detail chunk view."
    ),
)
async def get_file_chunks(
    collection_name: str,
    source: str = Query(..., description="Source file identifier to fetch chunks for"),
) -> FileChunksResponse:
    """Get all chunks for a single source file."""
    try:
        service = _get_dashboard_service()
        stats = await service.async_get_collection_stats(collection_name)
        if not stats.get("exists", False):
            raise HTTPException(
                status_code=404,
                detail={"error": f"Collection '{collection_name}' not found"},
            )
        
        chunks = await service.async_get_file_chunks(source, collection_name)
        if not chunks:
            raise HTTPException(
                status_code=404,
                detail={"error": f"No chunks found for source '{source}'"},
            )
        
        items = [FileChunkSchema(**c) for c in chunks]
        return FileChunksResponse(source=source, chunks=items, total=len(items))
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to get chunks for '%s' in '%s': %s",
            source, collection_name, e, exc_info=True,
        )
        raise HTTPException(status_code=500, detail={"error": str(e)})


# ==================== Delete a Single File ====================


@router.delete(
    "/{collection_name}/files",
    response_model=DeleteFileResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Delete a single file",
    description=(
        "Deletes every chunk belonging to one source file within a "
        "collection. Idempotent — deleting a file that no longer exists "
        "returns deleted_chunks=0."
    ),
)
async def delete_file(
    collection_name: str,
    source: str = Query(..., description="Source file identifier to delete"),
) -> DeleteFileResponse:
    """Delete all chunks of a single source file."""
    try:
        service = _get_dashboard_service()
        stats = await service.async_get_collection_stats(collection_name)
        if not stats.get("exists", False):
            raise HTTPException(
                status_code=404,
                detail={"error": f"Collection '{collection_name}' not found"},
            )
        
        deleted = await service.async_delete_file(source, collection_name)
        return DeleteFileResponse(
            success=True, source=source, deleted_chunks=deleted,
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to delete file '%s' in '%s': %s",
            source, collection_name, e, exc_info=True,
        )
        raise HTTPException(status_code=500, detail={"error": str(e)})
