"""
Data layer for RAGService.

Contains:
- VectorDB: Vector database abstractions and implementations
- Embeddings: Embedding model abstractions and implementations
- DocumentProcessors: Document loading and text splitting utilities
- services: High-level services for VectorDB operations and asset uploads
"""

from RAGService.Data.VectorDB import (
    VectorDBProvider,
    VectorDBConfig,
    BaseVectorDB,
    VectorDBFactory,
    SearchResult,
    DocumentChunk,
    CollectionInfo,
    MetadataFilter,
    FilterOperator,
)

from RAGService.Data.Embeddings import (
    EmbeddingProvider,
    EmbeddingConfig,
    BaseEmbeddings,
    EmbeddingsFactory,
)

from RAGService.Data.services import (
    VectorDBService,
    AssetUploadService,
    AssetUploadConfig,
    get_vectordb_service,
    get_asset_upload_service,
)

from RAGService.Data.DocumentProcessors import (
    SmartChunker,
    SmartChunkerConfig,
    ChunkMetadata,
)

__all__ = [
    # VectorDB
    "VectorDBProvider",
    "VectorDBConfig",
    "BaseVectorDB",
    "VectorDBFactory",
    "SearchResult",
    "DocumentChunk",
    "CollectionInfo",
    "MetadataFilter",
    "FilterOperator",
    # Embeddings
    "EmbeddingProvider",
    "EmbeddingConfig",
    "BaseEmbeddings",
    "EmbeddingsFactory",
    # Services
    "VectorDBService",
    "AssetUploadService",
    "AssetUploadConfig",
    "get_vectordb_service",
    "get_asset_upload_service",
    # Smart Chunking
    "SmartChunker",
    "SmartChunkerConfig",
    "ChunkMetadata",
]
