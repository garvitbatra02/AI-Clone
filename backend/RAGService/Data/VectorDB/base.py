"""
VectorDB Base Module

This module provides the core abstractions for vector database operations:
- VectorDBProvider: Enum of supported vector database providers
- VectorDBConfig: Configuration dataclass for vector database connections
- BaseVectorDB: Abstract base class defining the vector database interface
- SearchResult: Response model for search operations
- DocumentChunk: Model for document chunks with content and metadata
- CollectionInfo: Model for collection metadata
- MetadataFilter: Filter model for structured queries
- FilterOperator: Enum of filter operators
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Union


class VectorDBProvider(str, Enum):
    """Supported vector database providers."""
    QDRANT = "qdrant"
    PINECONE = "pinecone"
    CHROMA = "chroma"
    WEAVIATE = "weaviate"
    MILVUS = "milvus"
    PGVECTOR = "pgvector"


class DistanceMetric(str, Enum):
    """Distance metrics for vector similarity."""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"


class FilterOperator(str, Enum):
    """Operators for metadata filtering."""
    EQ = "eq"           # Equal
    NE = "ne"           # Not equal
    GT = "gt"           # Greater than
    GTE = "gte"         # Greater than or equal
    LT = "lt"           # Less than
    LTE = "lte"         # Less than or equal
    IN = "in"           # In list
    NIN = "nin"         # Not in list
    CONTAINS = "contains"  # Contains substring
    EXISTS = "exists"   # Field exists


@dataclass
class MetadataFilter:
    """
    Filter for metadata-based queries.
    
    Attributes:
        field: The metadata field to filter on
        operator: The filter operator to apply
        value: The value to compare against
    """
    field: str
    operator: FilterOperator
    value: Any
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert filter to dictionary representation."""
        return {
            "field": self.field,
            "operator": self.operator.value,
            "value": self.value
        }


@dataclass
class MetadataFilterGroup:
    """
    Group of metadata filters with AND/OR logic.
    
    Attributes:
        filters: List of MetadataFilter or nested MetadataFilterGroup
        operator: 'and' or 'or' to combine filters
    """
    filters: List[Union[MetadataFilter, "MetadataFilterGroup"]]
    operator: str = "and"  # "and" or "or"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert filter group to dictionary representation."""
        return {
            "operator": self.operator,
            "filters": [f.to_dict() for f in self.filters]
        }


@dataclass
class DocumentChunk:
    """
    A chunk of a document with content and metadata.
    
    This is the unified chunk model used across the entire RAG pipeline:
    document loading → chunking → embedding → vector storage.
    
    Attributes:
        content: The text content of the chunk
        metadata: Arbitrary metadata associated with the chunk
        id: Unique identifier for the chunk (auto-generated if not provided)
        embedding: Optional pre-computed embedding vector
        source: Source file or URL of the document
        chunk_index: Index of this chunk within the source document
        start_char: Starting character index in the original document
        end_char: Ending character index in the original document
    """
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None
    embedding: Optional[List[float]] = None
    source: Optional[str] = None
    chunk_index: Optional[int] = None
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    
    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())
        # Add source and chunk_index to metadata if provided
        if self.source and "source" not in self.metadata:
            self.metadata["source"] = self.source
        if self.chunk_index is not None and "chunk_index" not in self.metadata:
            self.metadata["chunk_index"] = self.chunk_index


@dataclass
class SearchResult:
    """
    Result from a vector similarity search.
    
    Attributes:
        id: Unique identifier of the matched document
        content: The text content of the matched document
        score: Similarity score (higher is more similar for cosine/dot product)
        metadata: Metadata associated with the document
        embedding: Optional embedding vector of the document
    """
    id: str
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None


@dataclass
class CollectionInfo:
    """
    Information about a vector database collection.
    
    Attributes:
        name: Name of the collection
        vector_count: Number of vectors in the collection
        dimension: Dimension of vectors in the collection
        distance_metric: Distance metric used for similarity
        metadata: Additional collection metadata
    """
    name: str
    vector_count: int
    dimension: int
    distance_metric: DistanceMetric
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorDBConfig:
    """
    Configuration for vector database connection.
    
    Attributes:
        provider: The vector database provider to use
        collection_name: Name of the collection/index to use
        embedding_dimension: Dimension of the embedding vectors
        url: Connection URL for the database (optional for in-memory)
        api_key: API key for authentication (if required)
        distance_metric: Distance metric for similarity search
        timeout: Connection timeout in seconds
        prefer_grpc: Whether to prefer gRPC over HTTP (provider-specific)
        extra_config: Additional provider-specific configuration
    """
    provider: VectorDBProvider
    collection_name: str
    embedding_dimension: int
    url: Optional[str] = None
    api_key: Optional[str] = None
    distance_metric: DistanceMetric = DistanceMetric.COSINE
    timeout: int = 30
    prefer_grpc: bool = True
    extra_config: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def for_qdrant(
        cls,
        collection_name: str,
        embedding_dimension: int,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        distance_metric: DistanceMetric = DistanceMetric.COSINE,
        **kwargs
    ) -> "VectorDBConfig":
        """Create configuration for Qdrant."""
        return cls(
            provider=VectorDBProvider.QDRANT,
            collection_name=collection_name,
            embedding_dimension=embedding_dimension,
            url=url,
            api_key=api_key,
            distance_metric=distance_metric,
            **kwargs
        )


class BaseVectorDB(ABC):
    """
    Abstract base class for vector database implementations.
    
    All vector database providers must implement this interface to ensure
    consistent behavior across different backends.
    
    The class provides both synchronous and asynchronous methods for all
    operations to support different use cases.
    """
    
    def __init__(self, config: VectorDBConfig):
        """
        Initialize the vector database connection.
        
        Args:
            config: Configuration for the database connection
        """
        self.config = config
        self._initialize_client()
    
    @abstractmethod
    def _initialize_client(self) -> None:
        """Initialize the database client. Must be implemented by subclasses."""
        pass
    
    @property
    @abstractmethod
    def provider(self) -> VectorDBProvider:
        """Return the provider type for this implementation."""
        pass
    
    # ==================== Collection Operations ====================
    
    @abstractmethod
    def create_collection(
        self,
        collection_name: Optional[str] = None,
        dimension: Optional[int] = None,
        distance_metric: Optional[DistanceMetric] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Create a new collection.
        
        Args:
            collection_name: Name of the collection (uses config if not provided)
            dimension: Vector dimension (uses config if not provided)
            distance_metric: Distance metric (uses config if not provided)
            metadata: Optional metadata for the collection
            
        Returns:
            True if collection was created successfully
        """
        pass
    
    @abstractmethod
    async def async_create_collection(
        self,
        collection_name: Optional[str] = None,
        dimension: Optional[int] = None,
        distance_metric: Optional[DistanceMetric] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Async version of create_collection."""
        pass
    
    @abstractmethod
    def delete_collection(self, collection_name: Optional[str] = None) -> bool:
        """
        Delete a collection.
        
        Args:
            collection_name: Name of the collection (uses config if not provided)
            
        Returns:
            True if collection was deleted successfully
        """
        pass
    
    @abstractmethod
    async def async_delete_collection(self, collection_name: Optional[str] = None) -> bool:
        """Async version of delete_collection."""
        pass
    
    @abstractmethod
    def collection_exists(self, collection_name: Optional[str] = None) -> bool:
        """
        Check if a collection exists.
        
        Args:
            collection_name: Name of the collection (uses config if not provided)
            
        Returns:
            True if collection exists
        """
        pass
    
    @abstractmethod
    async def async_collection_exists(self, collection_name: Optional[str] = None) -> bool:
        """Async version of collection_exists."""
        pass
    
    @abstractmethod
    def list_collections(self) -> List[str]:
        """
        List all collections in the database.
        
        Returns:
            List of collection names
        """
        pass
    
    @abstractmethod
    async def async_list_collections(self) -> List[str]:
        """Async version of list_collections."""
        pass
    
    @abstractmethod
    def get_collection_info(self, collection_name: Optional[str] = None) -> CollectionInfo:
        """
        Get information about a collection.
        
        Args:
            collection_name: Name of the collection (uses config if not provided)
            
        Returns:
            CollectionInfo with collection details
        """
        pass
    
    @abstractmethod
    async def async_get_collection_info(self, collection_name: Optional[str] = None) -> CollectionInfo:
        """Async version of get_collection_info."""
        pass
    
    # ==================== Document CRUD Operations ====================
    
    @abstractmethod
    def add_documents(
        self,
        documents: Sequence[DocumentChunk],
        collection_name: Optional[str] = None,
        batch_size: int = 100
    ) -> List[str]:
        """
        Add documents to the collection.
        
        Args:
            documents: List of DocumentChunk objects to add
            collection_name: Target collection (uses config if not provided)
            batch_size: Number of documents to add per batch
            
        Returns:
            List of IDs of added documents
        """
        pass
    
    @abstractmethod
    async def async_add_documents(
        self,
        documents: Sequence[DocumentChunk],
        collection_name: Optional[str] = None,
        batch_size: int = 100
    ) -> List[str]:
        """Async version of add_documents."""
        pass
    
    @abstractmethod
    def get_by_ids(
        self,
        ids: List[str],
        collection_name: Optional[str] = None,
        include_embeddings: bool = False
    ) -> List[DocumentChunk]:
        """
        Retrieve documents by their IDs.
        
        Args:
            ids: List of document IDs to retrieve
            collection_name: Target collection (uses config if not provided)
            include_embeddings: Whether to include embedding vectors
            
        Returns:
            List of DocumentChunk objects
        """
        pass
    
    @abstractmethod
    async def async_get_by_ids(
        self,
        ids: List[str],
        collection_name: Optional[str] = None,
        include_embeddings: bool = False
    ) -> List[DocumentChunk]:
        """Async version of get_by_ids."""
        pass
    
    @abstractmethod
    def update_documents(
        self,
        documents: Sequence[DocumentChunk],
        collection_name: Optional[str] = None
    ) -> List[str]:
        """
        Update existing documents in the collection.
        
        Args:
            documents: List of DocumentChunk objects with IDs to update
            collection_name: Target collection (uses config if not provided)
            
        Returns:
            List of IDs of updated documents
        """
        pass
    
    @abstractmethod
    async def async_update_documents(
        self,
        documents: Sequence[DocumentChunk],
        collection_name: Optional[str] = None
    ) -> List[str]:
        """Async version of update_documents."""
        pass
    
    @abstractmethod
    def delete_by_ids(
        self,
        ids: List[str],
        collection_name: Optional[str] = None
    ) -> bool:
        """
        Delete documents by their IDs.
        
        Args:
            ids: List of document IDs to delete
            collection_name: Target collection (uses config if not provided)
            
        Returns:
            True if deletion was successful
        """
        pass
    
    @abstractmethod
    async def async_delete_by_ids(
        self,
        ids: List[str],
        collection_name: Optional[str] = None
    ) -> bool:
        """Async version of delete_by_ids."""
        pass
    
    @abstractmethod
    def delete_by_filter(
        self,
        filters: Union[MetadataFilter, MetadataFilterGroup],
        collection_name: Optional[str] = None
    ) -> int:
        """
        Delete documents matching the filter criteria.
        
        Args:
            filters: Metadata filter or filter group
            collection_name: Target collection (uses config if not provided)
            
        Returns:
            Number of documents deleted
        """
        pass
    
    @abstractmethod
    async def async_delete_by_filter(
        self,
        filters: Union[MetadataFilter, MetadataFilterGroup],
        collection_name: Optional[str] = None
    ) -> int:
        """Async version of delete_by_filter."""
        pass
    
    # ==================== Search Operations ====================
    
    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        k: int = 10,
        filters: Optional[Union[MetadataFilter, MetadataFilterGroup]] = None,
        collection_name: Optional[str] = None,
        score_threshold: Optional[float] = None
    ) -> List[SearchResult]:
        """
        Search for similar documents by embedding vector.
        
        Args:
            query_embedding: The query embedding vector
            k: Number of results to return
            filters: Optional metadata filters
            collection_name: Target collection (uses config if not provided)
            score_threshold: Minimum score threshold for results
            
        Returns:
            List of SearchResult objects ordered by similarity
        """
        pass
    
    @abstractmethod
    async def async_search(
        self,
        query_embedding: List[float],
        k: int = 10,
        filters: Optional[Union[MetadataFilter, MetadataFilterGroup]] = None,
        collection_name: Optional[str] = None,
        score_threshold: Optional[float] = None
    ) -> List[SearchResult]:
        """Async version of search."""
        pass
    
    @abstractmethod
    def similarity_search_with_score(
        self,
        query_embedding: List[float],
        k: int = 10,
        filters: Optional[Union[MetadataFilter, MetadataFilterGroup]] = None,
        collection_name: Optional[str] = None
    ) -> List[tuple[DocumentChunk, float]]:
        """
        Search for similar documents and return with scores.
        
        Args:
            query_embedding: The query embedding vector
            k: Number of results to return
            filters: Optional metadata filters
            collection_name: Target collection (uses config if not provided)
            
        Returns:
            List of (DocumentChunk, score) tuples
        """
        pass
    
    @abstractmethod
    async def async_similarity_search_with_score(
        self,
        query_embedding: List[float],
        k: int = 10,
        filters: Optional[Union[MetadataFilter, MetadataFilterGroup]] = None,
        collection_name: Optional[str] = None
    ) -> List[tuple[DocumentChunk, float]]:
        """Async version of similarity_search_with_score."""
        pass
    
    # ==================== Utility Operations ====================
    
    @abstractmethod
    def count(self, collection_name: Optional[str] = None) -> int:
        """
        Count the number of documents in the collection.
        
        Args:
            collection_name: Target collection (uses config if not provided)
            
        Returns:
            Number of documents in the collection
        """
        pass
    
    @abstractmethod
    async def async_count(self, collection_name: Optional[str] = None) -> int:
        """Async version of count."""
        pass
    
    def _get_collection_name(self, collection_name: Optional[str] = None) -> str:
        """Get the collection name, falling back to config if not provided."""
        return collection_name or self.config.collection_name
    
    def _get_dimension(self, dimension: Optional[int] = None) -> int:
        """Get the dimension, falling back to config if not provided."""
        return dimension or self.config.embedding_dimension
    
    def _get_distance_metric(self, distance_metric: Optional[DistanceMetric] = None) -> DistanceMetric:
        """Get the distance metric, falling back to config if not provided."""
        return distance_metric or self.config.distance_metric
