"""
Embeddings Module

Provides abstractions and implementations for embedding model operations.
"""

from RAGService.Data.Embeddings.base import (
    BaseEmbeddings,
    EmbeddingConfig,
    EmbeddingInputType,
    EmbeddingProvider,
)
from RAGService.Data.Embeddings.factory import EmbeddingsFactory
from RAGService.Data.Embeddings.registry import (
    get_provider_class,
    is_provider_available,
    list_available_providers,
    register_provider,
)

__all__ = [
    # Base classes and models
    "BaseEmbeddings",
    "EmbeddingConfig",
    "EmbeddingProvider",
    "EmbeddingInputType",
    # Factory
    "EmbeddingsFactory",
    # Registry functions
    "get_provider_class",
    "register_provider",
    "list_available_providers",
    "is_provider_available",
]
