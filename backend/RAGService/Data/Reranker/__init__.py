"""
Reranker Package

Provides reranking capabilities for search results using cross-encoder models.
Reranking improves retrieval quality by re-scoring vector search results with
a more powerful model that reads query and document together.

Typical two-stage retrieval:
1. Vector search: fast, returns top_k candidates
2. Reranker: precise, re-scores to top_n final results
"""

from RAGService.Data.Reranker.base import (
    BaseReranker,
    RerankerConfig,
    RerankerProvider,
    AllKeysFailedError,
)
from RAGService.Data.Reranker.cohere_reranker import CohereReranker

__all__ = [
    "BaseReranker",
    "RerankerConfig",
    "RerankerProvider",
    "AllKeysFailedError",
    "CohereReranker",
]
