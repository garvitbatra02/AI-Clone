"""
Base Reranker abstraction.

Defines the contract that all reranker implementations must follow,
providing a consistent interface for reranking search results.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Any
from enum import Enum
import logging

from shared.key_rotation import KeyRotationMixin, AllKeysFailedError  # noqa: F401

logger = logging.getLogger(__name__)


class RerankerProvider(str, Enum):
    """Supported reranker providers."""
    COHERE = "cohere"


@dataclass
class RerankerConfig:
    """
    Configuration for Reranker instances.
    
    Attributes:
        model: The reranker model identifier
        top_n: Number of top results to return after reranking
        max_tokens_per_doc: Maximum tokens per document for reranking
        api_key: Optional single API key (backward compat shorthand)
        api_keys: Optional list of API keys for automatic rotation
        max_retries: Maximum retry sweeps across all keys on failure
        timeout: Request timeout in seconds
    """
    model: str = "rerank-v3.5"
    top_n: int = 5
    max_tokens_per_doc: int = 4096
    api_key: Optional[str] = None
    api_keys: Optional[list[str]] = None
    max_retries: int = 2
    timeout: int = 60


class BaseReranker(KeyRotationMixin, ABC):
    """
    Abstract base class for all reranker implementations.
    
    Inherits multi-key rotation, client caching, and retry logic from
    ``KeyRotationMixin``.  API keys are fully encapsulated.
    
    Subclasses must implement:
    - _do_rerank(): Synchronous raw reranking
    - _do_arerank(): Asynchronous raw reranking
    - _initialize_client(api_key): Create provider client(s)
    - provider: Property returning the provider type
    """
    
    ENV_VAR_NAME: str = ""  # Subclasses must override
    
    def __init__(self, config: Optional[RerankerConfig] = None):
        """
        Initialize the reranker.
        
        Args:
            config: RerankerConfig instance (uses defaults if omitted)
        """
        self.config = config or RerankerConfig()
        # Merge single api_key into api_keys list for backward compat
        api_keys = self.config.api_keys or (
            [self.config.api_key] if self.config.api_key else None
        )
        self._init_key_rotation(api_keys, self.config.max_retries)
    
    @abstractmethod
    def _initialize_client(self, api_key: str) -> None:
        """
        Initialize the provider-specific client with given API key.
        
        Must store the client as ``self._client`` (and optionally
        ``self._async_client``).  The mixin caches both automatically.
        """
        pass
    
    @property
    @abstractmethod
    def provider(self) -> RerankerProvider:
        """Return the provider type for this reranker."""
        pass
    
    # ==================== Abstract raw API methods ====================
    
    @abstractmethod
    def _do_rerank(
        self,
        query: str,
        results: list,
        top_n: Optional[int] = None,
    ) -> list:
        """
        Raw: rerank search results by relevance to the query.
        Resilience is handled by the public ``rerank`` wrapper.
        """
        pass
    
    @abstractmethod
    async def _do_arerank(
        self,
        query: str,
        results: list,
        top_n: Optional[int] = None,
    ) -> list:
        """
        Async raw: rerank search results by relevance to the query.
        Resilience is handled by the public ``arerank`` wrapper.
        """
        pass
    
    # ==================== Public methods with resilience ====================
    
    def rerank(
        self,
        query: str,
        results: list,
        top_n: Optional[int] = None,
    ) -> list:
        """
        Rerank search results with automatic key rotation and retry.
        
        Args:
            query: The search query
            results: List of SearchResult objects to rerank
            top_n: Override number of results to return (uses config if None)
            
        Returns:
            Reranked list of SearchResult objects with updated scores
        """
        if not results:
            return []
        return self._execute_with_rotation(
            operation=lambda: self._do_rerank(query, results, top_n),
            is_valid=lambda r: r is not None,
            service_label=self.provider.value,
            model_label=self.config.model,
        )
    
    async def arerank(
        self,
        query: str,
        results: list,
        top_n: Optional[int] = None,
    ) -> list:
        """
        Async rerank search results with automatic key rotation and retry.
        
        Args:
            query: The search query
            results: List of SearchResult objects to rerank
            top_n: Override number of results to return (uses config if None)
            
        Returns:
            Reranked list of SearchResult objects with updated scores
        """
        if not results:
            return []
        return await self._execute_with_rotation_async(
            operation=lambda: self._do_arerank(query, results, top_n),
            is_valid=lambda r: r is not None,
            service_label=self.provider.value,
            model_label=self.config.model,
        )
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.config.model})"
