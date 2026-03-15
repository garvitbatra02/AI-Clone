"""
Dashboard Service — Orchestration layer for the Asset Upload Dashboard.

This service sits between the HTTP routes and the core AssetUploadService.
It handles two concerns that the core service shouldn't own:

1. **Temp-file lifecycle** for HTTP uploads:
   Receives a FastAPI UploadFile, saves it to a temp file, delegates to
   the core service with the Path, then cleans up.

2. **Local-path passthrough**:
   For local testing / CLI workflows, directly delegates to the core
   service with no temp-file overhead.

The core AssetUploadService (RAGService.Data.services.asset_upload_service)
remains unchanged and continues to accept plain file paths.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from RAGService.Data.services.asset_upload_service import (
    AssetUploadConfig,
    AssetUploadService,
    UploadPreview,
    UploadResult,
    get_asset_upload_service,
)

logger = logging.getLogger(__name__)


class DashboardService:
    """
    Thin orchestration wrapper around AssetUploadService.
    
    Provides two families of methods:
      - ``*_uploaded_file()``  — for HTTP multipart uploads (handles temp files)
      - ``*_local_*()``        — for local path testing (direct delegation)
    
    The class never duplicates chunking / embedding logic; it always
    delegates to the core service.
    """
    
    def __init__(
        self,
        core_service: Optional[AssetUploadService] = None,
        config: Optional[AssetUploadConfig] = None,
    ):
        """
        Args:
            core_service: Pre-built AssetUploadService (preferred).
            config: Config forwarded to get_asset_upload_service() if
                    core_service is not supplied.
        """
        self._core = core_service or get_asset_upload_service(config)
    
    # ------------------------------------------------------------------
    #  Properties — expose core service for routes that need direct access
    # ------------------------------------------------------------------
    
    @property
    def core(self) -> AssetUploadService:
        """Access the underlying AssetUploadService."""
        return self._core
    
    # ------------------------------------------------------------------
    #  HTTP-upload helpers (temp-file lifecycle)
    # ------------------------------------------------------------------
    
    def _save_upload_to_temp(
        self,
        file_content: bytes,
        original_filename: str,
    ) -> Path:
        """
        Persist uploaded bytes to a temporary file, preserving the
        original extension so the loader factory can auto-detect the type.
        
        Returns the Path to the temp file.  Caller is responsible for
        cleanup via ``_cleanup_temp()``.
        """
        suffix = Path(original_filename).suffix  # e.g. ".pdf"
        tmp = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
            prefix="asset_upload_",
        )
        try:
            tmp.write(file_content)
        finally:
            tmp.close()
        
        return Path(tmp.name)
    
    @staticmethod
    def _cleanup_temp(path: Path) -> None:
        """Silently remove a temp file."""
        try:
            path.unlink(missing_ok=True)
        except Exception as exc:
            logger.warning("Failed to clean up temp file %s: %s", path, exc)
    
    # ---------- preview ----------
    
    async def preview_uploaded_file(
        self,
        file_content: bytes,
        original_filename: str,
        metadata: Optional[Dict[str, Any]] = None,
        max_preview_chunks: int = 5,
    ) -> UploadPreview:
        """
        Preview an HTTP-uploaded file (dry-run, no embed/store).
        
        Args:
            file_content: Raw bytes read from the UploadFile
            original_filename: Original filename (used for extension detection)
            metadata: Extra metadata to attach
            max_preview_chunks: How many chunk samples to include
            
        Returns:
            UploadPreview with file info and sample chunks
        """
        tmp_path = self._save_upload_to_temp(file_content, original_filename)
        try:
            preview = await self._core.async_preview_file(
                tmp_path,
                metadata=metadata,
                max_preview_chunks=max_preview_chunks,
            )
            # Override file_name to show the real upload name, not the tmp name
            preview.file_name = original_filename
            preview.metadata["original_filename"] = original_filename
            return preview
        finally:
            self._cleanup_temp(tmp_path)
    
    # ---------- upload ----------
    
    async def upload_uploaded_file(
        self,
        file_content: bytes,
        original_filename: str,
        collection_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> UploadResult:
        """
        Upload an HTTP-uploaded file to the vector DB.
        
        Args:
            file_content: Raw bytes read from the UploadFile
            original_filename: Original filename
            collection_name: Target collection (None = default)
            metadata: Extra metadata
            
        Returns:
            UploadResult from the core service
        """
        tmp_path = self._save_upload_to_temp(file_content, original_filename)
        try:
            file_metadata = {**(metadata or {}), "original_filename": original_filename}
            result = await self._core.async_upload_file(
                tmp_path,
                collection_name=collection_name,
                metadata=file_metadata,
            )
            # Rewrite source so it shows the real name, not the tmp path
            result.source = original_filename
            return result
        finally:
            self._cleanup_temp(tmp_path)
    
    # ------------------------------------------------------------------
    #  Local-path passthrough (for testing & CLI usage)
    # ------------------------------------------------------------------
    
    def preview_local_file(
        self,
        file_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
        max_preview_chunks: int = 5,
        **loader_kwargs,
    ) -> UploadPreview:
        """
        Preview a local file (sync, no temp-file overhead).
        Delegates directly to the core service.
        """
        return self._core.preview_file(
            file_path,
            metadata=metadata,
            max_preview_chunks=max_preview_chunks,
            **loader_kwargs,
        )
    
    async def async_preview_local_file(
        self,
        file_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
        max_preview_chunks: int = 5,
        **loader_kwargs,
    ) -> UploadPreview:
        """Async version of preview_local_file."""
        return await self._core.async_preview_file(
            file_path,
            metadata=metadata,
            max_preview_chunks=max_preview_chunks,
            **loader_kwargs,
        )
    
    def upload_local_file(
        self,
        file_path: Union[str, Path],
        collection_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **loader_kwargs,
    ) -> UploadResult:
        """Upload a local file (sync). Delegates directly to the core service."""
        return self._core.upload_file(
            file_path,
            collection_name=collection_name,
            metadata=metadata,
            **loader_kwargs,
        )
    
    async def async_upload_local_file(
        self,
        file_path: Union[str, Path],
        collection_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **loader_kwargs,
    ) -> UploadResult:
        """Upload a local file (async). Delegates directly to the core service."""
        return await self._core.async_upload_file(
            file_path,
            collection_name=collection_name,
            metadata=metadata,
            **loader_kwargs,
        )
    
    # ---------- directory ----------
    
    def upload_local_directory(
        self,
        directory_path: Union[str, Path],
        collection_name: Optional[str] = None,
        recursive: bool = True,
        extensions: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **loader_kwargs,
    ) -> List[UploadResult]:
        """Upload all supported files from a local directory (sync)."""
        return self._core.upload_directory(
            directory_path,
            collection_name=collection_name,
            recursive=recursive,
            extensions=extensions,
            metadata=metadata,
            **loader_kwargs,
        )
    
    async def async_upload_local_directory(
        self,
        directory_path: Union[str, Path],
        collection_name: Optional[str] = None,
        recursive: bool = True,
        extensions: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **loader_kwargs,
    ) -> List[UploadResult]:
        """Upload all supported files from a local directory (async)."""
        return await self._core.async_upload_directory(
            directory_path,
            collection_name=collection_name,
            recursive=recursive,
            extensions=extensions,
            metadata=metadata,
            **loader_kwargs,
        )
    
    # ---------- text ----------
    
    async def async_upload_text(
        self,
        text: str,
        collection_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None,
    ) -> UploadResult:
        """Upload raw text to the vector DB."""
        return await self._core.async_upload_text(
            text,
            collection_name=collection_name,
            metadata=metadata,
            source=source,
        )
    
    async def async_upload_texts(
        self,
        texts: List[str],
        collection_name: Optional[str] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        sources: Optional[List[str]] = None,
    ) -> UploadResult:
        """Upload multiple text strings as separate documents."""
        return await self._core.async_upload_texts(
            texts,
            collection_name=collection_name,
            metadatas=metadatas,
            sources=sources,
        )
    
    # ------------------------------------------------------------------
    #  Collection management (delegates to core)
    # ------------------------------------------------------------------
    
    async def async_list_collections(self) -> List[str]:
        return await self._core.async_list_collections()
    
    async def async_get_collection_stats(
        self, collection_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        return await self._core.async_get_collection_stats(collection_name)
    
    async def async_create_collection(self, name: str) -> Dict[str, Any]:
        return await self._core.async_create_collection(name)
    
    async def async_delete_collection(self, name: str) -> bool:
        return await self._core.async_delete_collection(name)


# ---------------------------------------------------------------------------
#  Singleton
# ---------------------------------------------------------------------------

_dashboard_service_instance: Optional[DashboardService] = None


def get_dashboard_service(
    config: Optional[AssetUploadConfig] = None,
    reset: bool = False,
) -> DashboardService:
    """
    Get the global DashboardService singleton.
    
    On first call (or when *reset* is True), creates a new instance.
    The underlying AssetUploadService singleton is shared with any
    other consumer that calls ``get_asset_upload_service()``.
    
    Args:
        config: Forwarded to the core service on first init / reset.
        reset: Force re-creation.
        
    Returns:
        DashboardService instance.
    """
    global _dashboard_service_instance
    
    if _dashboard_service_instance is None or reset:
        core = get_asset_upload_service(config, reset=reset)
        _dashboard_service_instance = DashboardService(core_service=core)
    
    return _dashboard_service_instance
