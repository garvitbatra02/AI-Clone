"""
Retrieval Service

Orchestrates the retrieval pipeline: vector search → rerank → format context.
This is the core read-side service of the RAG pipeline.

Sits between VectorDBService (raw search) and RAGService (LLM generation),
providing a clean retrieval interface that returns formatted context strings
ready for injection into LLM prompts.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from RAGService.Data.VectorDB.base import (
    MetadataFilter,
    MetadataFilterGroup,
    SearchResult,
)
from RAGService.Data.services.vectordb_service import (
    VectorDBService,
    VectorDBServiceConfig,
    get_vectordb_service,
)
from prompts import CONTEXT_HEADER, CONTEXT_CHUNK_TEMPLATE, CONTEXT_SEPARATOR

logger = logging.getLogger(__name__)


# ==================== Data Models ====================


@dataclass
class RetrievalConfig:
    """
    Configuration for the RetrievalService.
    
    Attributes:
        top_k: Number of candidates to fetch from vector search (over-fetch for reranker)
        score_threshold: Minimum similarity score from vector search
        rerank_enabled: Whether to rerank results using a cross-encoder model
        rerank_model: Reranker model name
        rerank_top_n: Number of results to keep after reranking
        context_template: Template for formatting each chunk in the context string.
                         Supports {i} (1-based index), {content}, {score}, {source}, {id}.
        context_separator: Separator between formatted chunks
        context_header: Header prepended to the context string
    """
    top_k: int = 20
    score_threshold: Optional[float] = 0.3
    rerank_enabled: bool = True
    rerank_model: str = "rerank-v3.5"
    rerank_top_n: int = 5
    context_template: str = CONTEXT_CHUNK_TEMPLATE
    context_separator: str = CONTEXT_SEPARATOR
    context_header: str = CONTEXT_HEADER
    
    @classmethod
    def from_env(cls) -> "RetrievalConfig":
        """Create configuration from environment variables."""
        return cls(
            top_k=int(os.environ.get("RETRIEVAL_TOP_K", "20")),
            score_threshold=float(os.environ.get("RETRIEVAL_SCORE_THRESHOLD", "0.3")),
            rerank_enabled=os.environ.get("RETRIEVAL_RERANK_ENABLED", "true").lower() == "true",
            rerank_model=os.environ.get("RETRIEVAL_RERANK_MODEL", "rerank-v3.5"),
            rerank_top_n=int(os.environ.get("RETRIEVAL_RERANK_TOP_N", "5")),
        )


@dataclass
class RetrievalResult:
    """
    Result from the retrieval pipeline.
    
    Attributes:
        context_str: Formatted context string ready for LLM prompt injection
        source_chunks: The ranked SearchResult objects used to build the context
        query: The original query string
        reranked: Whether results were reranked
        total_candidates: Number of candidates from vector search before reranking
    """
    context_str: str
    source_chunks: List[SearchResult]
    query: str
    reranked: bool = False
    total_candidates: int = 0


# ==================== Service ====================


class RetrievalService:
    """
    Orchestrates the retrieval pipeline for RAG.
    
    Pipeline:
    1. Vector search via VectorDBService (returns top_k candidates)
    2. Optional reranking via CohereReranker (narrows to top_n)
    3. Context formatting (produces a string for LLM prompt injection)
    
    The reranker is initialized lazily and degrades gracefully — if the
    Cohere API key is missing, reranking is skipped with a warning.
    
    Example:
        service = RetrievalService()
        result = service.retrieve(
            query="What is machine learning?",
            collection_name="my_docs",
        )
        print(result.context_str)  # Formatted context for LLM
        print(result.source_chunks)  # Ranked SearchResult objects
    """
    
    def __init__(
        self,
        config: Optional[RetrievalConfig] = None,
        vectordb_service: Optional[VectorDBService] = None,
    ):
        """
        Initialize the retrieval service.
        
        Args:
            config: Retrieval configuration (uses defaults/env if omitted)
            vectordb_service: Pre-configured VectorDBService (uses singleton if omitted)
        """
        self.config = config or RetrievalConfig.from_env()
        self._vectordb_service = vectordb_service
        self._reranker = None
        self._reranker_initialized = False
    
    @property
    def vectordb_service(self) -> VectorDBService:
        """Get or create the VectorDB service (lazy)."""
        if self._vectordb_service is None:
            self._vectordb_service = get_vectordb_service()
        return self._vectordb_service
    
    def _get_reranker(self):
        """
        Lazily initialize the reranker.
        
        Returns None if reranking is disabled or if the Cohere API key
        is not available (graceful degradation).
        """
        if self._reranker_initialized:
            return self._reranker
        
        self._reranker_initialized = True
        
        if not self.config.rerank_enabled:
            logger.info("Reranking is disabled in config")
            return None
        
        try:
            from RAGService.Data.Reranker.cohere_reranker import CohereReranker
            from RAGService.Data.Reranker.base import RerankerConfig
            
            reranker_config = RerankerConfig(
                model=self.config.rerank_model,
                top_n=self.config.rerank_top_n,
            )
            self._reranker = CohereReranker(config=reranker_config)
            logger.info(f"Reranker initialized: {self._reranker}")
        except ValueError as e:
            logger.warning(
                f"Could not initialize reranker (missing API key?): {e}. "
                f"Falling back to vector-search-only mode."
            )
            self._reranker = None
        except ImportError as e:
            logger.warning(
                f"Could not import reranker: {e}. "
                f"Falling back to vector-search-only mode."
            )
            self._reranker = None
        
        return self._reranker
    
    def _format_context(self, chunks: List[SearchResult]) -> str:
        """
        Format search results into a context string for LLM prompt injection.
        
        Args:
            chunks: Ranked list of SearchResult objects
            
        Returns:
            Formatted context string
        """
        if not chunks:
            return ""
        
        formatted_parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.metadata.get("source", "unknown")
            formatted = self.config.context_template.format(
                i=i,
                content=chunk.content,
                score=chunk.score,
                source=source,
                id=chunk.id,
            )
            formatted_parts.append(formatted)
        
        context_body = self.config.context_separator.join(formatted_parts)
        return f"{self.config.context_header}{context_body}"
    
    # ==================== Core Retrieval Methods ====================
    
    def retrieve(
        self,
        query: str,
        collection_name: Optional[str] = None,
        filters: Optional[Union[MetadataFilter, MetadataFilterGroup]] = None,
        top_k: Optional[int] = None,
        rerank_top_n: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> RetrievalResult:
        """
        Execute the full retrieval pipeline: search → rerank → format.
        
        Args:
            query: The search query
            collection_name: Target collection (uses VectorDBService default if None)
            filters: Optional metadata filters for vector search
            top_k: Override for number of vector search candidates
            rerank_top_n: Override for number of results after reranking
            score_threshold: Override for minimum similarity score
            
        Returns:
            RetrievalResult with context string and source chunks
        """
        effective_top_k = top_k or self.config.top_k
        effective_threshold = score_threshold if score_threshold is not None else self.config.score_threshold
        
        # Stage 1: Vector search
        logger.info(
            f"Retrieving for query: '{query[:80]}...' "
            f"(top_k={effective_top_k}, collection={collection_name})"
        )
        
        candidates = self.vectordb_service.search(
            query=query,
            k=effective_top_k,
            filters=filters,
            collection_name=collection_name,
            score_threshold=effective_threshold,
        )
        
        total_candidates = len(candidates)
        logger.info(f"Vector search returned {total_candidates} candidates")
        
        if not candidates:
            return RetrievalResult(
                context_str="",
                source_chunks=[],
                query=query,
                reranked=False,
                total_candidates=0,
            )
        
        # Stage 2: Reranking (optional)
        reranked = False
        reranker = self._get_reranker()
        
        if reranker is not None and len(candidates) > 1:
            effective_rerank_top_n = rerank_top_n or self.config.rerank_top_n
            try:
                candidates = reranker.rerank(
                    query=query,
                    results=candidates,
                    top_n=effective_rerank_top_n,
                )
                reranked = True
                logger.info(f"Reranked to {len(candidates)} results")
            except Exception as e:
                logger.warning(
                    f"Reranking failed: {e}. Using vector search results directly."
                )
        
        # Stage 3: Format context
        context_str = self._format_context(candidates)
        
        return RetrievalResult(
            context_str=context_str,
            source_chunks=candidates,
            query=query,
            reranked=reranked,
            total_candidates=total_candidates,
        )
    
    async def aretrieve(
        self,
        query: str,
        collection_name: Optional[str] = None,
        filters: Optional[Union[MetadataFilter, MetadataFilterGroup]] = None,
        top_k: Optional[int] = None,
        rerank_top_n: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> RetrievalResult:
        """
        Async version of the full retrieval pipeline.
        
        Args:
            query: The search query
            collection_name: Target collection
            filters: Optional metadata filters
            top_k: Override for vector search candidates
            rerank_top_n: Override for results after reranking
            score_threshold: Override for minimum similarity score
            
        Returns:
            RetrievalResult with context string and source chunks
        """
        effective_top_k = top_k or self.config.top_k
        effective_threshold = score_threshold if score_threshold is not None else self.config.score_threshold
        
        # Stage 1: Async vector search
        logger.info(
            f"Async retrieving for query: '{query[:80]}...' "
            f"(top_k={effective_top_k}, collection={collection_name})"
        )
        
        candidates = await self.vectordb_service.async_search(
            query=query,
            k=effective_top_k,
            filters=filters,
            collection_name=collection_name,
            score_threshold=effective_threshold,
        )
        
        total_candidates = len(candidates)
        logger.info(f"Async vector search returned {total_candidates} candidates")
        
        if not candidates:
            return RetrievalResult(
                context_str="",
                source_chunks=[],
                query=query,
                reranked=False,
                total_candidates=0,
            )
        
        # Stage 2: Async reranking (optional)
        reranked = False
        reranker = self._get_reranker()
        
        if reranker is not None and len(candidates) > 1:
            effective_rerank_top_n = rerank_top_n or self.config.rerank_top_n
            try:
                candidates = await reranker.arerank(
                    query=query,
                    results=candidates,
                    top_n=effective_rerank_top_n,
                )
                reranked = True
                logger.info(f"Async reranked to {len(candidates)} results")
            except Exception as e:
                logger.warning(
                    f"Async reranking failed: {e}. Using vector search results directly."
                )
        
        # Stage 3: Format context
        context_str = self._format_context(candidates)
        
        return RetrievalResult(
            context_str=context_str,
            source_chunks=candidates,
            query=query,
            reranked=reranked,
            total_candidates=total_candidates,
        )
    
    # ==================== Utility Methods ====================
    
    def search_only(
        self,
        query: str,
        collection_name: Optional[str] = None,
        filters: Optional[Union[MetadataFilter, MetadataFilterGroup]] = None,
        k: int = 10,
        score_threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        """
        Perform vector search without reranking or context formatting.
        
        Useful for raw search results or debugging.
        
        Args:
            query: The search query
            collection_name: Target collection
            filters: Optional metadata filters
            k: Number of results
            score_threshold: Minimum similarity score
            
        Returns:
            List of SearchResult objects
        """
        return self.vectordb_service.search(
            query=query,
            k=k,
            filters=filters,
            collection_name=collection_name,
            score_threshold=score_threshold,
        )
    
    async def asearch_only(
        self,
        query: str,
        collection_name: Optional[str] = None,
        filters: Optional[Union[MetadataFilter, MetadataFilterGroup]] = None,
        k: int = 10,
        score_threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        """Async version of search_only."""
        return await self.vectordb_service.async_search(
            query=query,
            k=k,
            filters=filters,
            collection_name=collection_name,
            score_threshold=score_threshold,
        )


# ==================== Global Singleton ====================

_retrieval_service: Optional[RetrievalService] = None
_retrieval_lock = threading.Lock()


def get_retrieval_service(
    config: Optional[RetrievalConfig] = None,
    vectordb_service: Optional[VectorDBService] = None,
    force_new: bool = False,
) -> RetrievalService:
    """
    Get the global RetrievalService singleton.
    
    Args:
        config: Retrieval configuration (uses defaults if omitted)
        vectordb_service: Pre-configured VectorDBService (uses singleton if omitted)
        force_new: Force creation of a new instance
        
    Returns:
        RetrievalService singleton instance
    """
    global _retrieval_service
    
    with _retrieval_lock:
        if _retrieval_service is None or force_new:
            _retrieval_service = RetrievalService(
                config=config,
                vectordb_service=vectordb_service,
            )
        return _retrieval_service
