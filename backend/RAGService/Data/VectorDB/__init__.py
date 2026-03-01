"""
VectorDB Module

Provides abstractions and implementations for vector database operations.
"""

from RAGService.Data.VectorDB.base import (
    BaseVectorDB,
    CollectionInfo,
    DistanceMetric,
    DocumentChunk,
    FilterOperator,
    MetadataFilter,
    MetadataFilterGroup,
    SearchResult,
    VectorDBConfig,
    VectorDBProvider,
)
from RAGService.Data.VectorDB.factory import VectorDBFactory
from RAGService.Data.VectorDB.registry import (
    get_provider_class,
    is_provider_available,
    list_available_providers,
    register_provider,
)

__all__ = [
    # Base classes and models
    "BaseVectorDB",
    "VectorDBConfig",
    "VectorDBProvider",
    "DistanceMetric",
    "DocumentChunk",
    "SearchResult",
    "CollectionInfo",
    "MetadataFilter",
    "MetadataFilterGroup",
    "FilterOperator",
    # Factory
    "VectorDBFactory",
    # Registry functions
    "get_provider_class",
    "register_provider",
    "list_available_providers",
    "is_provider_available",
]
