"""
Cohere Reranker implementation using the direct Cohere SDK.

Uses Cohere's rerank endpoint to re-score search results using a
cross-encoder model for more accurate relevance ranking.

Supports models:
- rerank-v3.5: Single multilingual model (default)
- rerank-english-v3.0: English only
- rerank-multilingual-v3.0: Multilingual

API keys are loaded automatically from the COHERE_API_KEYS environment
variable (falls back to COHERE_API_KEY for backward compat).
Resilience features (key rotation, retry) are inherited from BaseReranker.
"""

import os
import logging
from typing import List, Optional

from RAGService.Data.Reranker.base import BaseReranker, RerankerConfig, RerankerProvider
from RAGService.Data.VectorDB.base import SearchResult

logger = logging.getLogger(__name__)


class CohereReranker(BaseReranker):
    """
    Cohere implementation of the Reranker interface.
    
    Uses cohere.ClientV2.rerank() to re-score search results with a
    cross-encoder model. This produces much more accurate relevance
    scores than embedding cosine similarity alone.
    
    Typical usage pattern (two-stage retrieval):
    1. Vector search returns top_k=20 candidates (fast, approximate)
    2. Reranker re-scores them and returns top_n=5 (slow, precise)
    
    API keys are loaded from COHERE_API_KEYS environment variable
    (falls back to COHERE_API_KEY for backward compat).
    Resilience features (key rotation, retry) are inherited from BaseReranker.
    
    Example:
        reranker = CohereReranker()
        reranked = reranker.rerank(
            query="What is machine learning?",
            results=search_results,
            top_n=5,
        )
    """
    
    ENV_VAR_NAME = "COHERE_API_KEYS"
    _FALLBACK_ENV_VAR = "COHERE_API_KEY"
    
    def _load_api_keys_from_env(self) -> list[str]:
        """Try COHERE_API_KEYS first, fall back to COHERE_API_KEY."""
        keys = super()._load_api_keys_from_env()
        if not keys:
            single = os.getenv(self._FALLBACK_ENV_VAR, "").strip()
            if single:
                keys = [single]
        return keys
    
    def _initialize_client(self, api_key: str) -> None:
        """Initialize the Cohere client for reranking with given API key."""
        try:
            from cohere import ClientV2, AsyncClientV2
        except ImportError:
            raise ImportError(
                "Cohere package is required for reranking. "
                "Install it with: pip install cohere>=5.0.0"
            )
        
        self._client = ClientV2(api_key=api_key, timeout=self.config.timeout)
        self._async_client = AsyncClientV2(api_key=api_key, timeout=self.config.timeout)
    
    @property
    def provider(self) -> RerankerProvider:
        """Return the provider type."""
        return RerankerProvider.COHERE
    
    def _do_rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_n: Optional[int] = None,
    ) -> List[SearchResult]:
        """
        Raw: rerank search results using Cohere's cross-encoder model.
        Resilience is handled by BaseReranker.rerank().
        
        Args:
            query: The search query
            results: List of SearchResult objects from vector search
            top_n: Number of top results to return (uses config default if None)
            
        Returns:
            Reranked list of SearchResult objects with updated relevance scores,
            sorted by relevance (highest first)
        """
        if not results:
            return []
        
        effective_top_n = top_n or self.config.top_n
        # Don't request more results than we have
        effective_top_n = min(effective_top_n, len(results))
        
        # Extract document texts for reranking
        documents = [r.content for r in results]
        
        logger.info(
            f"Reranking {len(documents)} documents with query: "
            f"'{query[:50]}...' (top_n={effective_top_n})"
        )
        
        response = self._client.rerank(
            model=self.config.model,
            query=query,
            documents=documents,
            top_n=effective_top_n,
            max_tokens_per_doc=self.config.max_tokens_per_doc,
        )
        
        # Map reranked results back to SearchResult objects with updated scores
        reranked_results = []
        for item in response.results:
            original = results[item.index]
            reranked_results.append(SearchResult(
                id=original.id,
                content=original.content,
                score=item.relevance_score,
                metadata={
                    **original.metadata,
                    "original_score": original.score,
                    "rerank_score": item.relevance_score,
                    "reranked": True,
                },
                embedding=original.embedding,
            ))
        
        logger.info(
            f"Reranking complete. Top score: {reranked_results[0].score:.4f}" 
            if reranked_results else "Reranking returned no results"
        )
        
        return reranked_results
    
    async def _do_arerank(
        self,
        query: str,
        results: List[SearchResult],
        top_n: Optional[int] = None,
    ) -> List[SearchResult]:
        """
        Async raw: rerank search results using Cohere's cross-encoder model.
        Resilience is handled by BaseReranker.arerank().
        
        Args:
            query: The search query
            results: List of SearchResult objects from vector search
            top_n: Number of top results to return (uses config default if None)
            
        Returns:
            Reranked list of SearchResult objects with updated relevance scores,
            sorted by relevance (highest first)
        """
        if not results:
            return []
        
        effective_top_n = top_n or self.config.top_n
        effective_top_n = min(effective_top_n, len(results))
        
        documents = [r.content for r in results]
        
        logger.info(
            f"Async reranking {len(documents)} documents with query: "
            f"'{query[:50]}...' (top_n={effective_top_n})"
        )
        
        response = await self._async_client.rerank(
            model=self.config.model,
            query=query,
            documents=documents,
            top_n=effective_top_n,
            max_tokens_per_doc=self.config.max_tokens_per_doc,
        )
        
        reranked_results = []
        for item in response.results:
            original = results[item.index]
            reranked_results.append(SearchResult(
                id=original.id,
                content=original.content,
                score=item.relevance_score,
                metadata={
                    **original.metadata,
                    "original_score": original.score,
                    "rerank_score": item.relevance_score,
                    "reranked": True,
                },
                embedding=original.embedding,
            ))
        
        logger.info(
            f"Async reranking complete. Top score: {reranked_results[0].score:.4f}" 
            if reranked_results else "Async reranking returned no results"
        )
        
        return reranked_results
