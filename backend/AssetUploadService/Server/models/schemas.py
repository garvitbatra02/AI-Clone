"""
Pydantic v2 request / response schemas for the Asset Upload Dashboard API.

Follows the same patterns used in ChatService.Server.models.schemas:
  - BaseModel with Field(...) descriptions
  - Config.json_schema_extra for OpenAPI examples
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


# ==================== Shared / Reusable ====================


class ErrorResponse(BaseModel):
    """Standard error envelope."""
    error: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional error context",
    )


# ==================== Collection Schemas ====================


class CollectionInfoSchema(BaseModel):
    """Detailed information about a single collection."""
    collection: str = Field(..., description="Collection name")
    exists: bool = Field(default=True, description="Whether the collection exists")
    vector_count: int = Field(default=0, description="Number of vectors stored")
    dimension: int = Field(default=0, description="Embedding dimension")
    distance_metric: str = Field(default="cosine", description="Distance metric used")

    class Config:
        json_schema_extra = {
            "example": {
                "collection": "my_docs",
                "exists": True,
                "vector_count": 1250,
                "dimension": 1024,
                "distance_metric": "cosine",
            }
        }


class CollectionListResponse(BaseModel):
    """Response listing all collections with their stats."""
    collections: List[CollectionInfoSchema] = Field(
        default_factory=list,
        description="Collections with metadata",
    )
    total: int = Field(default=0, description="Total number of collections")


class CreateCollectionRequest(BaseModel):
    """Request to create a new collection."""
    name: str = Field(
        ...,
        description="Name of the collection to create",
        min_length=1,
        max_length=128,
    )

    class Config:
        json_schema_extra = {"example": {"name": "research_papers"}}


class CreateCollectionResponse(BaseModel):
    """Result of creating a collection."""
    created: bool = Field(..., description="Whether the collection was newly created")
    collection: str = Field(..., description="Collection name")
    reason: Optional[str] = Field(
        default=None,
        description="Reason if not created (e.g. 'already_exists')",
    )


# ==================== Upload Preview Schemas ====================


class ChunkPreviewSchema(BaseModel):
    """A single chunk preview returned during a dry-run."""
    index: int = Field(..., description="Chunk index (0-based)")
    content_preview: str = Field(
        ..., description="First ~300 chars of the chunk content",
    )
    char_count: int = Field(..., description="Full character count of the chunk")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Chunk metadata",
    )


class UploadPreviewResponse(BaseModel):
    """
    Dry-run result — shows what an upload *would* produce without
    embedding or storing anything.
    """
    file_name: str = Field(..., description="Original file name")
    file_type: str = Field(..., description="Detected file type / extension")
    file_size_bytes: int = Field(..., description="File size in bytes")
    total_chunks: int = Field(..., description="Number of chunks that would be created")
    estimated_tokens: int = Field(
        ..., description="Rough token estimate (~4 chars/token)",
    )
    chunk_previews: List[ChunkPreviewSchema] = Field(
        default_factory=list,
        description="Sample of the first N chunks",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "file_name": "research.pdf",
                "file_type": "pdf",
                "file_size_bytes": 245000,
                "total_chunks": 42,
                "estimated_tokens": 18500,
                "chunk_previews": [
                    {
                        "index": 0,
                        "content_preview": "Introduction: Machine learning is a...",
                        "char_count": 950,
                        "metadata": {"source": "research.pdf", "page": 1},
                    }
                ],
                "metadata": {"source": "/tmp/research.pdf"},
            }
        }


# ==================== Upload Response Schemas ====================


class UploadFileResponse(BaseModel):
    """Result of a successful file / text upload."""
    success: bool = Field(..., description="Whether the upload succeeded")
    source: Optional[str] = Field(
        default=None, description="Source file name or identifier",
    )
    document_ids: List[str] = Field(
        default_factory=list, description="IDs of stored document chunks",
    )
    total_chunks: int = Field(default=0, description="Number of chunks created")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Upload metadata",
    )
    error: Optional[str] = Field(
        default=None, description="Error message if upload failed",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "source": "research.pdf",
                "document_ids": ["abc123", "def456"],
                "total_chunks": 42,
                "metadata": {"file_type": "pdf", "collection": "my_docs"},
            }
        }


class DirectoryUploadFileResult(BaseModel):
    """Result for one file within a directory upload."""
    success: bool
    source: Optional[str] = None
    total_chunks: int = 0
    error: Optional[str] = None


class DirectoryUploadResponse(BaseModel):
    """Result of uploading an entire directory."""
    total_files: int = Field(..., description="Number of files processed")
    successful: int = Field(..., description="Number of files uploaded successfully")
    failed: int = Field(..., description="Number of files that failed")
    results: List[DirectoryUploadFileResult] = Field(
        default_factory=list,
        description="Per-file results",
    )


# ==================== Request Bodies (JSON) ====================


class LocalFileRequest(BaseModel):
    """Request body for local file preview / upload."""
    file_path: str = Field(
        ..., description="Absolute path to the file on the server's filesystem",
    )
    collection_name: Optional[str] = Field(
        default=None,
        description="Target collection (uses default if omitted)",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional metadata",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "file_path": "/Users/me/docs/research.pdf",
                "collection_name": "research",
            }
        }


class LocalDirectoryRequest(BaseModel):
    """Request body for local directory upload."""
    directory_path: str = Field(
        ..., description="Absolute path to the directory",
    )
    collection_name: Optional[str] = Field(
        default=None, description="Target collection",
    )
    recursive: bool = Field(
        default=True, description="Whether to traverse subdirectories",
    )
    extensions: Optional[List[str]] = Field(
        default=None,
        description="Filter to these extensions (e.g. ['pdf', 'txt']).  None = all supported.",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional metadata for all files",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "directory_path": "/Users/me/docs",
                "collection_name": "my_docs",
                "recursive": True,
                "extensions": ["pdf", "txt"],
            }
        }


class UploadTextRequest(BaseModel):
    """Request body for raw text upload."""
    text: str = Field(
        ..., description="Text content to upload", min_length=1,
    )
    collection_name: Optional[str] = Field(
        default=None, description="Target collection",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Metadata to attach",
    )
    source: Optional[str] = Field(
        default=None,
        description="Source identifier (e.g. 'manual_input', 'clipboard')",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "text": "The quick brown fox jumps over the lazy dog.",
                "collection_name": "notes",
                "source": "manual_input",
            }
        }


class UploadTextsRequest(BaseModel):
    """Request body for batch text upload."""
    texts: List[str] = Field(
        ..., description="List of text strings", min_length=1,
    )
    collection_name: Optional[str] = Field(
        default=None, description="Target collection",
    )
    metadatas: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Per-text metadata (must match texts length if provided)",
    )
    sources: Optional[List[str]] = Field(
        default=None,
        description="Per-text source identifiers",
    )


class SupportedTypesResponse(BaseModel):
    """Response listing supported file extensions."""
    extensions: List[str] = Field(
        ..., description="List of supported file extensions (without dot)",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "extensions": ["txt", "md", "markdown", "json", "csv", "pdf", "docx"]
            }
        }
