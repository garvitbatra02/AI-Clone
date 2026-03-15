"""
RAGService - Retrieval Augmented Generation Service

This module provides a comprehensive RAG implementation with:
- Extensible Vector Database abstraction (Qdrant, Pinecone, Chroma, etc.)
- Pluggable Embedding models (Cohere, OpenAI, HuggingFace, etc.)
- Multi-format Document Processing (PDF, TXT, MD, JSON, CSV, DOCX)
- Asset upload utilities and application-facing CRUD services
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
    RetrievalService,
    RetrievalConfig,
    RetrievalResult,
    get_retrieval_service,
    RAGService,
    RAGConfig,
    RAGResponse,
    get_rag_service,
)

from RAGService.Data.DocumentProcessors import (
    SmartChunker,
    SmartChunkerConfig,
    ChunkMetadata,
)

from RAGService.Data.Reranker import (
    BaseReranker,
    RerankerConfig,
    RerankerProvider,
    CohereReranker,
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
    "RetrievalService",
    "RetrievalConfig",
    "RetrievalResult",
    "get_retrieval_service",
    "RAGService",
    "RAGConfig",
    "RAGResponse",
    "get_rag_service",
    # Smart Chunking
    "SmartChunker",
    "SmartChunkerConfig",
    "ChunkMetadata",
    # Reranker
    "BaseReranker",
    "RerankerConfig",
    "RerankerProvider",
    "CohereReranker",
]
