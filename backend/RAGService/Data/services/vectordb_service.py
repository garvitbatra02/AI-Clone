"""
VectorDB Service

High-level service for application-facing VectorDB operations.
Provides a unified interface for all CRUD and search operations.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

from RAGService.Data.VectorDB import (
    BaseVectorDB,
    CollectionInfo,
    DistanceMetric,
    DocumentChunk,
    MetadataFilter,
    MetadataFilterGroup,
    SearchResult,
    VectorDBFactory,
    VectorDBProvider,
)
from RAGService.Data.Embeddings import (
    BaseEmbeddings,
    EmbeddingConfig,
    EmbeddingInputType,
    EmbeddingsFactory,
    EmbeddingProvider,
)


@dataclass
class VectorDBServiceConfig:
    """
    Configuration for VectorDBService.
    
    Attributes:
        vectordb_provider: Vector database provider to use
        embedding_provider: Embedding model provider to use
        collection_name: Default collection name
        in_memory: Whether to use in-memory storage (True) or cloud (False)
        embedding_model: Embedding model name
        distance_metric: Distance metric for similarity
        auto_create_collection: Whether to auto-create collection if not exists
    """
    vectordb_provider: VectorDBProvider = VectorDBProvider.QDRANT
    embedding_provider: EmbeddingProvider = EmbeddingProvider.COHERE
    collection_name: str = "default"
    in_memory: bool = False
    embedding_model: Optional[str] = None
    distance_metric: DistanceMetric = DistanceMetric.COSINE
    auto_create_collection: bool = True
    
    @classmethod
    def from_env(cls, collection_name: str = "default") -> "VectorDBServiceConfig":
        """Create configuration from environment variables."""
        return cls(
            vectordb_provider=VectorDBProvider(
                os.environ.get("VECTORDB_PROVIDER", "qdrant")
            ),
            embedding_provider=EmbeddingProvider(
                os.environ.get("EMBEDDING_PROVIDER", "cohere")
            ),
            collection_name=collection_name,
            in_memory=os.environ.get("VECTORDB_IN_MEMORY", "false").lower() == "true",
            embedding_model=os.environ.get("EMBEDDING_MODEL"),
        )


class VectorDBService:
    """
    High-level service for VectorDB operations.
    
    Combines vector database and embedding model to provide a simple
    interface for storing and searching documents.
    
    Example:
        # Create service
        service = VectorDBService(
            config=VectorDBServiceConfig(
                collection_name="my_docs",
            )
        )
        
        # Add documents (auto-embeds)
        service.add_texts(["Hello world", "Foo bar"])
        
        # Search
        results = service.search("greeting")
        
        # Search with filters
        results = service.search(
            "greeting",
            filters=MetadataFilter(field="category", operator=FilterOperator.EQ, value="chat")
        )
    """
    
    def __init__(
        self,
        config: Optional[VectorDBServiceConfig] = None,
        vectordb: Optional[BaseVectorDB] = None,
        embeddings: Optional[BaseEmbeddings] = None
    ):
        """
        Initialize the VectorDB service.
        
        Args:
            config: Service configuration
            vectordb: Pre-configured VectorDB instance (optional)
            embeddings: Pre-configured embeddings instance (optional)
        """
        self.config = config or VectorDBServiceConfig.from_env()
        
        # Initialize embeddings first to get dimension
        if embeddings:
            self._embeddings = embeddings
        else:
            self._embeddings = self._create_embeddings()
        
        # Initialize vector database
        if vectordb:
            self._vectordb = vectordb
        else:
            self._vectordb = self._create_vectordb()
        
        # Auto-create collection if needed
        if self.config.auto_create_collection:
            self._ensure_collection_exists()
    
    def _create_embeddings(self) -> BaseEmbeddings:
        """Create embeddings instance from config."""
        if self.config.embedding_provider == EmbeddingProvider.COHERE:
            return EmbeddingsFactory.create_cohere(
                model_name=self.config.embedding_model or "embed-english-v3.0"
            )
        elif self.config.embedding_provider == EmbeddingProvider.OPENAI:
            return EmbeddingsFactory.create_openai(
                model_name=self.config.embedding_model or "text-embedding-3-small"
            )
        elif self.config.embedding_provider == EmbeddingProvider.HUGGINGFACE:
            return EmbeddingsFactory.create_huggingface(
                model_name=self.config.embedding_model or "sentence-transformers/all-mpnet-base-v2"
            )
        else:
            return EmbeddingsFactory.create_from_env(self.config.embedding_provider)
    
    def _create_vectordb(self) -> BaseVectorDB:
        """Create VectorDB instance from config."""
        return VectorDBFactory.create_from_env(
            provider=self.config.vectordb_provider,
            collection_name=self.config.collection_name,
            embedding_dimension=self._embeddings.dimension,
            distance_metric=self.config.distance_metric,
            in_memory=self.config.in_memory,
        )
    
    def _ensure_collection_exists(self) -> None:
        """Ensure the collection exists, creating if necessary."""
        if not self._vectordb.collection_exists():
            self._vectordb.create_collection()
    
    @property
    def vectordb(self) -> BaseVectorDB:
        """Access the underlying VectorDB instance."""
        return self._vectordb
    
    @property
    def embeddings(self) -> BaseEmbeddings:
        """Access the underlying embeddings instance."""
        return self._embeddings
    
    # ==================== Document Operations ====================
    
    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        collection_name: Optional[str] = None,
        batch_size: int = 100
    ) -> List[str]:
        """
        Add texts to the vector database.
        
        Automatically embeds the texts before adding.
        
        Args:
            texts: List of text strings to add
            metadatas: Optional list of metadata dicts (one per text)
            ids: Optional list of IDs (auto-generated if not provided)
            collection_name: Target collection (uses default if not provided)
            batch_size: Batch size for embedding and insertion
            
        Returns:
            List of document IDs
        """
        if not texts:
            return []
        
        # Generate embeddings
        embeddings = self._embeddings.embed_documents(
            texts,
            input_type=EmbeddingInputType.SEARCH_DOCUMENT,
            batch_size=batch_size
        )
        
        # Create DocumentChunks
        documents = []
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
            doc_id = ids[i] if ids and i < len(ids) else None
            
            doc = DocumentChunk(
                content=text,
                metadata=metadata,
                id=doc_id,
                embedding=embedding
            )
            documents.append(doc)
        
        return self._vectordb.add_documents(
            documents,
            collection_name=collection_name,
            batch_size=batch_size
        )
    
    async def async_add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        collection_name: Optional[str] = None,
        batch_size: int = 100
    ) -> List[str]:
        """Async version of add_texts."""
        if not texts:
            return []
        
        embeddings = await self._embeddings.async_embed_documents(
            texts,
            input_type=EmbeddingInputType.SEARCH_DOCUMENT,
            batch_size=batch_size
        )
        
        documents = []
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
            doc_id = ids[i] if ids and i < len(ids) else None
            
            doc = DocumentChunk(
                content=text,
                metadata=metadata,
                id=doc_id,
                embedding=embedding
            )
            documents.append(doc)
        
        return await self._vectordb.async_add_documents(
            documents,
            collection_name=collection_name,
            batch_size=batch_size
        )
    
    def add_documents(
        self,
        documents: Sequence[DocumentChunk],
        collection_name: Optional[str] = None,
        batch_size: int = 100
    ) -> List[str]:
        """
        Add pre-chunked documents to the vector database.
        
        If documents don't have embeddings, they will be generated.
        
        Args:
            documents: List of DocumentChunk objects
            collection_name: Target collection
            batch_size: Batch size for operations
            
        Returns:
            List of document IDs
        """
        # Check if embeddings are needed
        docs_needing_embeddings = [d for d in documents if d.embedding is None]
        
        if docs_needing_embeddings:
            texts = [d.content for d in docs_needing_embeddings]
            embeddings = self._embeddings.embed_documents(
                texts,
                input_type=EmbeddingInputType.SEARCH_DOCUMENT,
                batch_size=batch_size
            )
            
            for doc, embedding in zip(docs_needing_embeddings, embeddings):
                doc.embedding = embedding
        
        return self._vectordb.add_documents(
            documents,
            collection_name=collection_name,
            batch_size=batch_size
        )
    
    async def async_add_documents(
        self,
        documents: Sequence[DocumentChunk],
        collection_name: Optional[str] = None,
        batch_size: int = 100
    ) -> List[str]:
        """Async version of add_documents."""
        docs_needing_embeddings = [d for d in documents if d.embedding is None]
        
        if docs_needing_embeddings:
            texts = [d.content for d in docs_needing_embeddings]
            embeddings = await self._embeddings.async_embed_documents(
                texts,
                input_type=EmbeddingInputType.SEARCH_DOCUMENT,
                batch_size=batch_size
            )
            
            for doc, embedding in zip(docs_needing_embeddings, embeddings):
                doc.embedding = embedding
        
        return await self._vectordb.async_add_documents(
            documents,
            collection_name=collection_name,
            batch_size=batch_size
        )
    
    def get_by_ids(
        self,
        ids: List[str],
        collection_name: Optional[str] = None,
        include_embeddings: bool = False
    ) -> List[DocumentChunk]:
        """Get documents by their IDs."""
        return self._vectordb.get_by_ids(
            ids,
            collection_name=collection_name,
            include_embeddings=include_embeddings
        )
    
    async def async_get_by_ids(
        self,
        ids: List[str],
        collection_name: Optional[str] = None,
        include_embeddings: bool = False
    ) -> List[DocumentChunk]:
        """Async version of get_by_ids."""
        return await self._vectordb.async_get_by_ids(
            ids,
            collection_name=collection_name,
            include_embeddings=include_embeddings
        )
    
    def update_texts(
        self,
        ids: List[str],
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        collection_name: Optional[str] = None
    ) -> List[str]:
        """Update existing documents with new text."""
        embeddings = self._embeddings.embed_documents(
            texts,
            input_type=EmbeddingInputType.SEARCH_DOCUMENT
        )
        
        documents = []
        for i, (doc_id, text, embedding) in enumerate(zip(ids, texts, embeddings)):
            metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
            doc = DocumentChunk(
                id=doc_id,
                content=text,
                metadata=metadata,
                embedding=embedding
            )
            documents.append(doc)
        
        return self._vectordb.update_documents(documents, collection_name)
    
    async def async_update_texts(
        self,
        ids: List[str],
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        collection_name: Optional[str] = None
    ) -> List[str]:
        """Async version of update_texts."""
        embeddings = await self._embeddings.async_embed_documents(
            texts,
            input_type=EmbeddingInputType.SEARCH_DOCUMENT
        )
        
        documents = []
        for i, (doc_id, text, embedding) in enumerate(zip(ids, texts, embeddings)):
            metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
            doc = DocumentChunk(
                id=doc_id,
                content=text,
                metadata=metadata,
                embedding=embedding
            )
            documents.append(doc)
        
        return await self._vectordb.async_update_documents(documents, collection_name)
    
    def delete_by_ids(
        self,
        ids: List[str],
        collection_name: Optional[str] = None
    ) -> bool:
        """Delete documents by their IDs."""
        return self._vectordb.delete_by_ids(ids, collection_name)
    
    async def async_delete_by_ids(
        self,
        ids: List[str],
        collection_name: Optional[str] = None
    ) -> bool:
        """Async version of delete_by_ids."""
        return await self._vectordb.async_delete_by_ids(ids, collection_name)
    
    def delete_by_filter(
        self,
        filters: Union[MetadataFilter, MetadataFilterGroup],
        collection_name: Optional[str] = None
    ) -> int:
        """Delete documents matching a filter."""
        return self._vectordb.delete_by_filter(filters, collection_name)
    
    async def async_delete_by_filter(
        self,
        filters: Union[MetadataFilter, MetadataFilterGroup],
        collection_name: Optional[str] = None
    ) -> int:
        """Async version of delete_by_filter."""
        return await self._vectordb.async_delete_by_filter(filters, collection_name)
    
    # ==================== Search Operations ====================
    
    def search(
        self,
        query: str,
        k: int = 10,
        filters: Optional[Union[MetadataFilter, MetadataFilterGroup]] = None,
        collection_name: Optional[str] = None,
        score_threshold: Optional[float] = None
    ) -> List[SearchResult]:
        """
        Search for similar documents.
        
        Args:
            query: Search query text
            k: Number of results to return
            filters: Optional metadata filters
            collection_name: Target collection
            score_threshold: Minimum similarity score
            
        Returns:
            List of SearchResult objects
        """
        query_embedding = self._embeddings.embed_query(
            query,
            input_type=EmbeddingInputType.SEARCH_QUERY
        )
        
        return self._vectordb.search(
            query_embedding=query_embedding,
            k=k,
            filters=filters,
            collection_name=collection_name,
            score_threshold=score_threshold
        )
    
    async def async_search(
        self,
        query: str,
        k: int = 10,
        filters: Optional[Union[MetadataFilter, MetadataFilterGroup]] = None,
        collection_name: Optional[str] = None,
        score_threshold: Optional[float] = None
    ) -> List[SearchResult]:
        """Async version of search."""
        query_embedding = await self._embeddings.async_embed_query(
            query,
            input_type=EmbeddingInputType.SEARCH_QUERY
        )
        
        return await self._vectordb.async_search(
            query_embedding=query_embedding,
            k=k,
            filters=filters,
            collection_name=collection_name,
            score_threshold=score_threshold
        )
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 10,
        filters: Optional[Union[MetadataFilter, MetadataFilterGroup]] = None,
        collection_name: Optional[str] = None
    ) -> List[tuple[DocumentChunk, float]]:
        """Search and return documents with similarity scores."""
        query_embedding = self._embeddings.embed_query(
            query,
            input_type=EmbeddingInputType.SEARCH_QUERY
        )
        
        return self._vectordb.similarity_search_with_score(
            query_embedding=query_embedding,
            k=k,
            filters=filters,
            collection_name=collection_name
        )
    
    async def async_similarity_search_with_score(
        self,
        query: str,
        k: int = 10,
        filters: Optional[Union[MetadataFilter, MetadataFilterGroup]] = None,
        collection_name: Optional[str] = None
    ) -> List[tuple[DocumentChunk, float]]:
        """Async version of similarity_search_with_score."""
        query_embedding = await self._embeddings.async_embed_query(
            query,
            input_type=EmbeddingInputType.SEARCH_QUERY
        )
        
        return await self._vectordb.async_similarity_search_with_score(
            query_embedding=query_embedding,
            k=k,
            filters=filters,
            collection_name=collection_name
        )
    
    def search_by_vector(
        self,
        embedding: List[float],
        k: int = 10,
        filters: Optional[Union[MetadataFilter, MetadataFilterGroup]] = None,
        collection_name: Optional[str] = None,
        score_threshold: Optional[float] = None
    ) -> List[SearchResult]:
        """Search using a pre-computed embedding vector."""
        return self._vectordb.search(
            query_embedding=embedding,
            k=k,
            filters=filters,
            collection_name=collection_name,
            score_threshold=score_threshold
        )
    
    async def async_search_by_vector(
        self,
        embedding: List[float],
        k: int = 10,
        filters: Optional[Union[MetadataFilter, MetadataFilterGroup]] = None,
        collection_name: Optional[str] = None,
        score_threshold: Optional[float] = None
    ) -> List[SearchResult]:
        """Async version of search_by_vector."""
        return await self._vectordb.async_search(
            query_embedding=embedding,
            k=k,
            filters=filters,
            collection_name=collection_name,
            score_threshold=score_threshold
        )
    
    # ==================== Collection Operations ====================
    
    def create_collection(
        self,
        collection_name: str,
        dimension: Optional[int] = None,
        distance_metric: Optional[DistanceMetric] = None
    ) -> bool:
        """Create a new collection."""
        return self._vectordb.create_collection(
            collection_name=collection_name,
            dimension=dimension or self._embeddings.dimension,
            distance_metric=distance_metric
        )
    
    async def async_create_collection(
        self,
        collection_name: str,
        dimension: Optional[int] = None,
        distance_metric: Optional[DistanceMetric] = None
    ) -> bool:
        """Async version of create_collection."""
        return await self._vectordb.async_create_collection(
            collection_name=collection_name,
            dimension=dimension or self._embeddings.dimension,
            distance_metric=distance_metric
        )
    
    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection."""
        return self._vectordb.delete_collection(collection_name)
    
    async def async_delete_collection(self, collection_name: str) -> bool:
        """Async version of delete_collection."""
        return await self._vectordb.async_delete_collection(collection_name)
    
    def list_collections(self) -> List[str]:
        """List all collections."""
        return self._vectordb.list_collections()
    
    async def async_list_collections(self) -> List[str]:
        """Async version of list_collections."""
        return await self._vectordb.async_list_collections()
    
    def collection_exists(self, collection_name: Optional[str] = None) -> bool:
        """Check if a collection exists."""
        return self._vectordb.collection_exists(collection_name)
    
    async def async_collection_exists(self, collection_name: Optional[str] = None) -> bool:
        """Async version of collection_exists."""
        return await self._vectordb.async_collection_exists(collection_name)
    
    def get_collection_info(self, collection_name: Optional[str] = None) -> CollectionInfo:
        """Get collection information."""
        return self._vectordb.get_collection_info(collection_name)
    
    async def async_get_collection_info(self, collection_name: Optional[str] = None) -> CollectionInfo:
        """Async version of get_collection_info."""
        return await self._vectordb.async_get_collection_info(collection_name)
    
    def count(self, collection_name: Optional[str] = None) -> int:
        """Count documents in collection."""
        return self._vectordb.count(collection_name)
    
    async def async_count(self, collection_name: Optional[str] = None) -> int:
        """Async version of count."""
        return await self._vectordb.async_count(collection_name)


# Global singleton instance
_vectordb_service_instance: Optional[VectorDBService] = None


def get_vectordb_service(
    config: Optional[VectorDBServiceConfig] = None,
    reset: bool = False
) -> VectorDBService:
    """
    Get the global VectorDBService singleton.
    
    Args:
        config: Service configuration (only used on first call or reset)
        reset: Whether to reset the singleton
        
    Returns:
        VectorDBService instance
    """
    global _vectordb_service_instance
    
    if _vectordb_service_instance is None or reset:
        _vectordb_service_instance = VectorDBService(config)
    
    return _vectordb_service_instance
