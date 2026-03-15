"""
Upload routes for the Asset Upload Dashboard.

Prefix: /assets/uploads

Provides two parallel families of endpoints:
  • HTTP multipart  — POST /preview, POST /file
    (for browser / dashboard uploads)
  • Local-path JSON  — POST /preview/local, POST /file/local, POST /directory/local
    (for local testing / CLI without needing multipart)
  • Text            — POST /text, POST /texts
  • Utility         — GET  /supported-types
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from AssetUploadService.Server.models.schemas import (
    DirectoryUploadFileResult,
    DirectoryUploadResponse,
    ErrorResponse,
    LocalDirectoryRequest,
    LocalFileRequest,
    SupportedTypesResponse,
    UploadFileResponse,
    UploadPreviewResponse,
    UploadTextRequest,
    UploadTextsRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/assets/uploads", tags=["Asset Uploads"])


def _get_dashboard_service():
    """Lazy import to avoid circular imports at module level."""
    from AssetUploadService.services.dashboard_service import get_dashboard_service
    return get_dashboard_service()


def _upload_result_to_response(result) -> UploadFileResponse:
    """Convert a core UploadResult dataclass to the Pydantic response."""
    return UploadFileResponse(
        success=result.success,
        source=result.source,
        document_ids=result.document_ids,
        total_chunks=result.total_chunks,
        metadata=result.metadata,
        error=result.error,
    )


def _preview_to_response(preview) -> UploadPreviewResponse:
    """Convert a core UploadPreview dataclass to the Pydantic response."""
    return UploadPreviewResponse(
        file_name=preview.file_name,
        file_type=preview.file_type,
        file_size_bytes=preview.file_size_bytes,
        total_chunks=preview.total_chunks,
        estimated_tokens=preview.estimated_tokens,
        chunk_previews=preview.chunk_previews,
        metadata=preview.metadata,
    )


# ==================== Supported Types ====================


@router.get(
    "/supported-types",
    response_model=SupportedTypesResponse,
    summary="List supported file types",
    description="Returns file extensions that the upload service can process.",
)
async def supported_types() -> SupportedTypesResponse:
    """Get list of supported file extensions."""
    from RAGService.Data.services.asset_upload_service import AssetUploadService
    return SupportedTypesResponse(
        extensions=AssetUploadService.get_supported_file_types(),
    )


# ====================================================================
#  HTTP Multipart Uploads  (browser / dashboard)
# ====================================================================


@router.post(
    "/preview",
    response_model=UploadPreviewResponse,
    summary="Preview a file upload (dry-run)",
    description=(
        "Accepts a multipart file upload and returns a preview of what the "
        "upload would produce (chunk count, file type, size, sample chunks). "
        "Nothing is embedded or stored."
    ),
)
async def preview_upload(
    file: UploadFile = File(..., description="File to preview"),
) -> UploadPreviewResponse:
    """Preview an uploaded file without committing."""
    try:
        service = _get_dashboard_service()
        content = await file.read()
        
        preview = await service.preview_uploaded_file(
            file_content=content,
            original_filename=file.filename or "unknown",
        )
        return _preview_to_response(preview)
    
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail={"error": str(e)})
    except ValueError as e:
        raise HTTPException(status_code=400, detail={"error": str(e)})
    except Exception as e:
        logger.error("Preview failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail={"error": str(e)})


@router.post(
    "/file",
    response_model=UploadFileResponse,
    summary="Upload a file",
    description=(
        "Accepts a multipart file upload, auto-detects the file type, "
        "chunks, embeds, and stores it in the specified (or default) collection."
    ),
)
async def upload_file(
    file: UploadFile = File(..., description="File to upload"),
    collection_name: Optional[str] = Form(
        default=None, description="Target collection name",
    ),
) -> UploadFileResponse:
    """Upload a file to the vector database."""
    try:
        service = _get_dashboard_service()
        content = await file.read()
        
        result = await service.upload_uploaded_file(
            file_content=content,
            original_filename=file.filename or "unknown",
            collection_name=collection_name,
        )
        
        if not result.success:
            raise HTTPException(
                status_code=500,
                detail={"error": result.error or "Upload failed"},
            )
        
        return _upload_result_to_response(result)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("File upload failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail={"error": str(e)})


# ====================================================================
#  Local-Path Endpoints  (for testing / CLI)
# ====================================================================


@router.post(
    "/preview/local",
    response_model=UploadPreviewResponse,
    summary="Preview a local file (dry-run)",
    description=(
        "Preview a file by its local filesystem path. "
        "Useful for testing without needing multipart upload."
    ),
)
async def preview_local_file(request: LocalFileRequest) -> UploadPreviewResponse:
    """Preview a local file without committing."""
    try:
        path = Path(request.file_path)
        if not path.exists():
            raise HTTPException(
                status_code=404,
                detail={"error": f"File not found: {request.file_path}"},
            )
        
        service = _get_dashboard_service()
        preview = await service.async_preview_local_file(
            file_path=path,
            metadata=request.metadata,
        )
        return _preview_to_response(preview)
    
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail={"error": str(e)})
    except Exception as e:
        logger.error("Local preview failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail={"error": str(e)})


@router.post(
    "/file/local",
    response_model=UploadFileResponse,
    summary="Upload a local file",
    description=(
        "Upload a file by its local filesystem path. "
        "Useful for testing without needing multipart upload."
    ),
)
async def upload_local_file(request: LocalFileRequest) -> UploadFileResponse:
    """Upload a file from the local filesystem."""
    try:
        path = Path(request.file_path)
        if not path.exists():
            raise HTTPException(
                status_code=404,
                detail={"error": f"File not found: {request.file_path}"},
            )
        
        service = _get_dashboard_service()
        result = await service.async_upload_local_file(
            file_path=path,
            collection_name=request.collection_name,
            metadata=request.metadata,
        )
        
        if not result.success:
            raise HTTPException(
                status_code=500,
                detail={"error": result.error or "Upload failed"},
            )
        
        return _upload_result_to_response(result)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Local file upload failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail={"error": str(e)})


@router.post(
    "/directory/local",
    response_model=DirectoryUploadResponse,
    summary="Upload a local directory",
    description=(
        "Upload all supported files from a local directory. "
        "Optionally filter by extension and traverse subdirectories."
    ),
)
async def upload_local_directory(request: LocalDirectoryRequest) -> DirectoryUploadResponse:
    """Upload all supported files from a local directory."""
    try:
        path = Path(request.directory_path)
        if not path.is_dir():
            raise HTTPException(
                status_code=404,
                detail={"error": f"Directory not found: {request.directory_path}"},
            )
        
        service = _get_dashboard_service()
        results = await service.async_upload_local_directory(
            directory_path=path,
            collection_name=request.collection_name,
            recursive=request.recursive,
            extensions=request.extensions,
            metadata=request.metadata,
        )
        
        file_results = [
            DirectoryUploadFileResult(
                success=r.success,
                source=r.source,
                total_chunks=r.total_chunks,
                error=r.error,
            )
            for r in results
        ]
        successful = sum(1 for r in results if r.success)
        
        return DirectoryUploadResponse(
            total_files=len(results),
            successful=successful,
            failed=len(results) - successful,
            results=file_results,
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Directory upload failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail={"error": str(e)})


# ====================================================================
#  Text Uploads
# ====================================================================


@router.post(
    "/text",
    response_model=UploadFileResponse,
    summary="Upload raw text",
    description="Upload a raw text string to the vector database.",
)
async def upload_text(request: UploadTextRequest) -> UploadFileResponse:
    """Upload raw text."""
    try:
        service = _get_dashboard_service()
        result = await service.async_upload_text(
            text=request.text,
            collection_name=request.collection_name,
            metadata=request.metadata,
            source=request.source,
        )
        
        if not result.success:
            raise HTTPException(
                status_code=500,
                detail={"error": result.error or "Upload failed"},
            )
        
        return _upload_result_to_response(result)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Text upload failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail={"error": str(e)})


@router.post(
    "/texts",
    response_model=UploadFileResponse,
    summary="Upload multiple texts",
    description="Upload multiple text strings as separate documents in a single batch.",
)
async def upload_texts(request: UploadTextsRequest) -> UploadFileResponse:
    """Upload multiple texts as a batch."""
    try:
        service = _get_dashboard_service()
        result = await service.async_upload_texts(
            texts=request.texts,
            collection_name=request.collection_name,
            metadatas=request.metadatas,
            sources=request.sources,
        )
        
        if not result.success:
            raise HTTPException(
                status_code=500,
                detail={"error": result.error or "Upload failed"},
            )
        
        return _upload_result_to_response(result)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Batch text upload failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail={"error": str(e)})
