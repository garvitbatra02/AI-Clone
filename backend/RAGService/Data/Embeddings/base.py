"""
Embeddings Base Module

This module provides the core abstractions for embedding model operations:
- EmbeddingProvider: Enum of supported embedding providers
- EmbeddingConfig: Configuration dataclass for embedding models
- BaseEmbeddings: Abstract base class defining the embeddings interface
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class EmbeddingProvider(str, Enum):
    """Supported embedding model providers."""
    COHERE = "cohere"
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    GOOGLE = "google"
    AZURE_OPENAI = "azure_openai"
    VOYAGE = "voyage"
    MISTRAL = "mistral"


class EmbeddingInputType(str, Enum):
    """
    Input types for embeddings (used by some providers like Cohere).
    
    Different input types can optimize embeddings for specific use cases.
    """
    SEARCH_DOCUMENT = "search_document"  # For documents to be searched
    SEARCH_QUERY = "search_query"        # For search queries
    CLASSIFICATION = "classification"     # For classification tasks
    CLUSTERING = "clustering"            # For clustering tasks


@dataclass
class EmbeddingConfig:
    """
    Configuration for embedding models.
    
    Attributes:
        provider: The embedding provider to use
        model_name: Name/ID of the embedding model
        api_key: API key for the provider
        dimension: Expected dimension of embeddings (provider-dependent)
        batch_size: Maximum batch size for embedding requests
        timeout: Request timeout in seconds
        truncate: How to handle texts exceeding max length ("START", "END", "NONE")
        extra_config: Additional provider-specific configuration
    """
    provider: EmbeddingProvider
    model_name: str
    api_key: Optional[str] = None
    dimension: Optional[int] = None
    batch_size: int = 96  # Cohere's default max batch size
    timeout: int = 60
    truncate: str = "END"
    extra_config: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def for_cohere(
        cls,
        api_key: Optional[str] = None,
        model_name: str = "embed-english-v3.0",
        **kwargs
    ) -> "EmbeddingConfig":
        """
        Create configuration for Cohere embeddings.
        
        Default model: embed-english-v3.0 (1024 dimensions)
        Other options:
        - embed-english-light-v3.0 (384 dimensions)
        - embed-multilingual-v3.0 (1024 dimensions)
        - embed-multilingual-light-v3.0 (384 dimensions)
        """
        # Dimension mapping for Cohere models
        dimension_map = {
            "embed-english-v3.0": 1024,
            "embed-english-light-v3.0": 384,
            "embed-multilingual-v3.0": 1024,
            "embed-multilingual-light-v3.0": 384,
        }
        
        return cls(
            provider=EmbeddingProvider.COHERE,
            model_name=model_name,
            api_key=api_key,
            dimension=dimension_map.get(model_name, 1024),
            **kwargs
        )
    
    @classmethod
    def for_openai(
        cls,
        api_key: Optional[str] = None,
        model_name: str = "text-embedding-3-small",
        **kwargs
    ) -> "EmbeddingConfig":
        """
        Create configuration for OpenAI embeddings.
        
        Default model: text-embedding-3-small (1536 dimensions)
        Other options:
        - text-embedding-3-large (3072 dimensions)
        - text-embedding-ada-002 (1536 dimensions)
        """
        dimension_map = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        
        return cls(
            provider=EmbeddingProvider.OPENAI,
            model_name=model_name,
            api_key=api_key,
            dimension=dimension_map.get(model_name, 1536),
            **kwargs
        )
    
    @classmethod
    def for_huggingface(
        cls,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        **kwargs
    ) -> "EmbeddingConfig":
        """
        Create configuration for HuggingFace embeddings.
        
        Default model: sentence-transformers/all-mpnet-base-v2 (768 dimensions)
        Other options:
        - sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
        - BAAI/bge-large-en-v1.5 (1024 dimensions)
        """
        dimension_map = {
            "sentence-transformers/all-mpnet-base-v2": 768,
            "sentence-transformers/all-MiniLM-L6-v2": 384,
            "BAAI/bge-large-en-v1.5": 1024,
            "BAAI/bge-base-en-v1.5": 768,
            "BAAI/bge-small-en-v1.5": 384,
        }
        
        return cls(
            provider=EmbeddingProvider.HUGGINGFACE,
            model_name=model_name,
            api_key=None,  # HuggingFace local models don't need API key
            dimension=dimension_map.get(model_name, 768),
            **kwargs
        )


class BaseEmbeddings(ABC):
    """
    Abstract base class for embedding model implementations.
    
    All embedding providers must implement this interface to ensure
    consistent behavior across different backends.
    
    The class provides both synchronous and asynchronous methods for
    embedding queries and documents.
    """
    
    def __init__(self, config: EmbeddingConfig):
        """
        Initialize the embedding model.
        
        Args:
            config: Configuration for the embedding model
        """
        self.config = config
        self._initialize_client()
    
    @abstractmethod
    def _initialize_client(self) -> None:
        """Initialize the embedding client. Must be implemented by subclasses."""
        pass
    
    @property
    @abstractmethod
    def provider(self) -> EmbeddingProvider:
        """Return the provider type for this implementation."""
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimension of embeddings produced by this model."""
        pass
    
    @abstractmethod
    def embed_query(
        self,
        text: str,
        input_type: Optional[EmbeddingInputType] = None
    ) -> List[float]:
        """
        Embed a single query text.
        
        Args:
            text: The text to embed
            input_type: Optional input type hint for the provider
            
        Returns:
            Embedding vector as list of floats
        """
        pass
    
    @abstractmethod
    async def async_embed_query(
        self,
        text: str,
        input_type: Optional[EmbeddingInputType] = None
    ) -> List[float]:
        """Async version of embed_query."""
        pass
    
    @abstractmethod
    def embed_documents(
        self,
        texts: List[str],
        input_type: Optional[EmbeddingInputType] = None,
        batch_size: Optional[int] = None
    ) -> List[List[float]]:
        """
        Embed multiple documents.
        
        Args:
            texts: List of texts to embed
            input_type: Optional input type hint for the provider
            batch_size: Override default batch size
            
        Returns:
            List of embedding vectors
        """
        pass
    
    @abstractmethod
    async def async_embed_documents(
        self,
        texts: List[str],
        input_type: Optional[EmbeddingInputType] = None,
        batch_size: Optional[int] = None
    ) -> List[List[float]]:
        """Async version of embed_documents."""
        pass
    
    def embed_text(
        self,
        text: str,
        is_query: bool = True
    ) -> List[float]:
        """
        Convenience method to embed text with appropriate input type.
        
        Args:
            text: The text to embed
            is_query: True for search queries, False for documents
            
        Returns:
            Embedding vector
        """
        input_type = (
            EmbeddingInputType.SEARCH_QUERY if is_query
            else EmbeddingInputType.SEARCH_DOCUMENT
        )
        return self.embed_query(text, input_type)
    
    async def async_embed_text(
        self,
        text: str,
        is_query: bool = True
    ) -> List[float]:
        """Async version of embed_text."""
        input_type = (
            EmbeddingInputType.SEARCH_QUERY if is_query
            else EmbeddingInputType.SEARCH_DOCUMENT
        )
        return await self.async_embed_query(text, input_type)
