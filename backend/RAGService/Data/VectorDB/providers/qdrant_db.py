"""
Qdrant Vector Database Implementation

Provides a complete implementation of BaseVectorDB for Qdrant,
supporting both synchronous and asynchronous operations.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Sequence, Union

from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointIdsList,
    PointStruct,
    Range,
    UpdateStatus,
    VectorParams,
)

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


class QdrantVectorDB(BaseVectorDB):
    """
    Qdrant implementation of the VectorDB interface.
    
    Supports three connection modes:
    1. In-memory: For testing and development (set in_memory=True in extra_config)
    2. Local file: Persistent local storage (set path in extra_config)
    3. Cloud/Server: Remote Qdrant instance (set url in config)
    
    Example:
        # In-memory (testing)
        config = VectorDBConfig.for_qdrant(
            collection_name="test",
            embedding_dimension=768,
            extra_config={"in_memory": True}
        )
        
        # Local file storage
        config = VectorDBConfig.for_qdrant(
            collection_name="test",
            embedding_dimension=768,
            extra_config={"path": "./qdrant_data"}
        )
        
        # Remote server
        config = VectorDBConfig.for_qdrant(
            collection_name="test",
            embedding_dimension=768,
            url="http://localhost:6333",
            api_key="your-api-key"  # Optional for Qdrant Cloud
        )
    """
    
    _client: QdrantClient
    _async_client: AsyncQdrantClient
    
    def _initialize_client(self) -> None:
        """Initialize Qdrant clients (sync and async)."""
        extra = self.config.extra_config
        
        # Determine connection mode
        in_memory = extra.get("in_memory", False)
        path = extra.get("path")
        
        if in_memory:
            # In-memory mode for testing
            self._client = QdrantClient(location=":memory:")
            self._async_client = AsyncQdrantClient(location=":memory:")
        elif path:
            # Local file storage
            self._client = QdrantClient(path=path)
            self._async_client = AsyncQdrantClient(path=path)
        elif self.config.url:
            # Remote server
            self._client = QdrantClient(
                url=self.config.url,
                api_key=self.config.api_key,
                timeout=self.config.timeout,
                prefer_grpc=self.config.prefer_grpc,
            )
            self._async_client = AsyncQdrantClient(
                url=self.config.url,
                api_key=self.config.api_key,
                timeout=self.config.timeout,
                prefer_grpc=self.config.prefer_grpc,
            )
        else:
            # Default to in-memory if nothing specified
            self._client = QdrantClient(location=":memory:")
            self._async_client = AsyncQdrantClient(location=":memory:")
    
    @property
    def provider(self) -> VectorDBProvider:
        return VectorDBProvider.QDRANT
    
    def _to_qdrant_distance(self, metric: DistanceMetric) -> Distance:
        """Convert DistanceMetric to Qdrant Distance."""
        mapping = {
            DistanceMetric.COSINE: Distance.COSINE,
            DistanceMetric.EUCLIDEAN: Distance.EUCLID,
            DistanceMetric.DOT_PRODUCT: Distance.DOT,
        }
        return mapping.get(metric, Distance.COSINE)
    
    def _from_qdrant_distance(self, distance: Distance) -> DistanceMetric:
        """Convert Qdrant Distance to DistanceMetric."""
        mapping = {
            Distance.COSINE: DistanceMetric.COSINE,
            Distance.EUCLID: DistanceMetric.EUCLIDEAN,
            Distance.DOT: DistanceMetric.DOT_PRODUCT,
        }
        return mapping.get(distance, DistanceMetric.COSINE)
    
    def _build_qdrant_filter(
        self,
        filters: Optional[Union[MetadataFilter, MetadataFilterGroup]]
    ) -> Optional[Filter]:
        """Convert MetadataFilter/Group to Qdrant Filter."""
        if filters is None:
            return None
        
        if isinstance(filters, MetadataFilter):
            condition = self._build_field_condition(filters)
            return Filter(must=[condition])
        
        # MetadataFilterGroup
        conditions = [
            self._build_field_condition(f) if isinstance(f, MetadataFilter)
            else self._build_qdrant_filter(f)
            for f in filters.filters
        ]
        
        if filters.operator == "and":
            return Filter(must=conditions)
        else:  # "or"
            return Filter(should=conditions)
    
    def _build_field_condition(self, f: MetadataFilter) -> FieldCondition:
        """Build a Qdrant FieldCondition from MetadataFilter."""
        field = f.field
        op = f.operator
        value = f.value
        
        if op == FilterOperator.EQ:
            return FieldCondition(key=field, match=MatchValue(value=value))
        elif op == FilterOperator.NE:
            # Qdrant doesn't have direct NE, use must_not in Filter
            return FieldCondition(key=field, match=MatchValue(value=value))
        elif op == FilterOperator.GT:
            return FieldCondition(key=field, range=Range(gt=value))
        elif op == FilterOperator.GTE:
            return FieldCondition(key=field, range=Range(gte=value))
        elif op == FilterOperator.LT:
            return FieldCondition(key=field, range=Range(lt=value))
        elif op == FilterOperator.LTE:
            return FieldCondition(key=field, range=Range(lte=value))
        elif op == FilterOperator.IN:
            return FieldCondition(key=field, match=MatchAny(any=value))
        elif op == FilterOperator.NIN:
            # Handle with must_not
            return FieldCondition(key=field, match=MatchAny(any=value))
        elif op == FilterOperator.CONTAINS:
            # Qdrant text match
            return FieldCondition(
                key=field,
                match=qdrant_models.MatchText(text=value)
            )
        elif op == FilterOperator.EXISTS:
            # Use IsNotNull condition
            return FieldCondition(
                key=field,
                match=qdrant_models.MatchExcept(**{"except": [None]})
            )
        else:
            raise ValueError(f"Unsupported filter operator: {op}")
    
    # ==================== Collection Operations ====================
    
    def create_collection(
        self,
        collection_name: Optional[str] = None,
        dimension: Optional[int] = None,
        distance_metric: Optional[DistanceMetric] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Create a new Qdrant collection."""
        name = self._get_collection_name(collection_name)
        dim = self._get_dimension(dimension)
        metric = self._get_distance_metric(distance_metric)
        
        self._client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=dim,
                distance=self._to_qdrant_distance(metric)
            )
        )
        return True
    
    async def async_create_collection(
        self,
        collection_name: Optional[str] = None,
        dimension: Optional[int] = None,
        distance_metric: Optional[DistanceMetric] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Async version of create_collection."""
        name = self._get_collection_name(collection_name)
        dim = self._get_dimension(dimension)
        metric = self._get_distance_metric(distance_metric)
        
        await self._async_client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=dim,
                distance=self._to_qdrant_distance(metric)
            )
        )
        return True
    
    def delete_collection(self, collection_name: Optional[str] = None) -> bool:
        """Delete a Qdrant collection."""
        name = self._get_collection_name(collection_name)
        return self._client.delete_collection(collection_name=name)
    
    async def async_delete_collection(self, collection_name: Optional[str] = None) -> bool:
        """Async version of delete_collection."""
        name = self._get_collection_name(collection_name)
        return await self._async_client.delete_collection(collection_name=name)
    
    def collection_exists(self, collection_name: Optional[str] = None) -> bool:
        """Check if a Qdrant collection exists."""
        name = self._get_collection_name(collection_name)
        return self._client.collection_exists(collection_name=name)
    
    async def async_collection_exists(self, collection_name: Optional[str] = None) -> bool:
        """Async version of collection_exists."""
        name = self._get_collection_name(collection_name)
        return await self._async_client.collection_exists(collection_name=name)
    
    def list_collections(self) -> List[str]:
        """List all Qdrant collections."""
        collections = self._client.get_collections()
        return [c.name for c in collections.collections]
    
    async def async_list_collections(self) -> List[str]:
        """Async version of list_collections."""
        collections = await self._async_client.get_collections()
        return [c.name for c in collections.collections]
    
    def get_collection_info(self, collection_name: Optional[str] = None) -> CollectionInfo:
        """Get information about a Qdrant collection."""
        name = self._get_collection_name(collection_name)
        info = self._client.get_collection(collection_name=name)
        
        # Extract vector config
        vectors_config = info.config.params.vectors
        if isinstance(vectors_config, VectorParams):
            dimension = vectors_config.size
            distance = vectors_config.distance
        else:
            # Named vectors - get the first one
            first_config = next(iter(vectors_config.values()))
            dimension = first_config.size
            distance = first_config.distance
        
        return CollectionInfo(
            name=name,
            vector_count=info.points_count or 0,
            dimension=dimension,
            distance_metric=self._from_qdrant_distance(distance),
            metadata={
                "status": str(info.status),
                "optimizer_status": str(info.optimizer_status),
                "indexed_vectors_count": info.indexed_vectors_count,
            }
        )
    
    async def async_get_collection_info(self, collection_name: Optional[str] = None) -> CollectionInfo:
        """Async version of get_collection_info."""
        name = self._get_collection_name(collection_name)
        info = await self._async_client.get_collection(collection_name=name)
        
        vectors_config = info.config.params.vectors
        if isinstance(vectors_config, VectorParams):
            dimension = vectors_config.size
            distance = vectors_config.distance
        else:
            first_config = next(iter(vectors_config.values()))
            dimension = first_config.size
            distance = first_config.distance
        
        return CollectionInfo(
            name=name,
            vector_count=info.points_count or 0,
            dimension=dimension,
            distance_metric=self._from_qdrant_distance(distance),
            metadata={
                "status": str(info.status),
                "optimizer_status": str(info.optimizer_status),
                "indexed_vectors_count": info.indexed_vectors_count,
            }
        )
    
    # ==================== Document CRUD Operations ====================
    
    def add_documents(
        self,
        documents: Sequence[DocumentChunk],
        collection_name: Optional[str] = None,
        batch_size: int = 100
    ) -> List[str]:
        """Add documents to Qdrant."""
        name = self._get_collection_name(collection_name)
        added_ids = []
        
        # Process in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            points = []
            
            for doc in batch:
                if doc.embedding is None:
                    raise ValueError(f"Document {doc.id} has no embedding. "
                                   "Embed documents before adding to VectorDB.")
                
                point = PointStruct(
                    id=doc.id,
                    vector=doc.embedding,
                    payload={
                        "content": doc.content,
                        **doc.metadata
                    }
                )
                points.append(point)
                added_ids.append(doc.id)
            
            self._client.upsert(
                collection_name=name,
                points=points,
                wait=True
            )
        
        return added_ids
    
    async def async_add_documents(
        self,
        documents: Sequence[DocumentChunk],
        collection_name: Optional[str] = None,
        batch_size: int = 100
    ) -> List[str]:
        """Async version of add_documents."""
        name = self._get_collection_name(collection_name)
        added_ids = []
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            points = []
            
            for doc in batch:
                if doc.embedding is None:
                    raise ValueError(f"Document {doc.id} has no embedding.")
                
                point = PointStruct(
                    id=doc.id,
                    vector=doc.embedding,
                    payload={
                        "content": doc.content,
                        **doc.metadata
                    }
                )
                points.append(point)
                added_ids.append(doc.id)
            
            await self._async_client.upsert(
                collection_name=name,
                points=points,
                wait=True
            )
        
        return added_ids
    
    def get_by_ids(
        self,
        ids: List[str],
        collection_name: Optional[str] = None,
        include_embeddings: bool = False
    ) -> List[DocumentChunk]:
        """Retrieve documents by ID from Qdrant."""
        name = self._get_collection_name(collection_name)
        
        results = self._client.retrieve(
            collection_name=name,
            ids=ids,
            with_payload=True,
            with_vectors=include_embeddings
        )
        
        documents = []
        for point in results:
            payload = point.payload or {}
            content = payload.pop("content", "")
            
            doc = DocumentChunk(
                id=str(point.id),
                content=content,
                metadata=payload,
                embedding=point.vector if include_embeddings else None
            )
            documents.append(doc)
        
        return documents
    
    async def async_get_by_ids(
        self,
        ids: List[str],
        collection_name: Optional[str] = None,
        include_embeddings: bool = False
    ) -> List[DocumentChunk]:
        """Async version of get_by_ids."""
        name = self._get_collection_name(collection_name)
        
        results = await self._async_client.retrieve(
            collection_name=name,
            ids=ids,
            with_payload=True,
            with_vectors=include_embeddings
        )
        
        documents = []
        for point in results:
            payload = point.payload or {}
            content = payload.pop("content", "")
            
            doc = DocumentChunk(
                id=str(point.id),
                content=content,
                metadata=payload,
                embedding=point.vector if include_embeddings else None
            )
            documents.append(doc)
        
        return documents
    
    def update_documents(
        self,
        documents: Sequence[DocumentChunk],
        collection_name: Optional[str] = None
    ) -> List[str]:
        """Update documents in Qdrant (upsert behavior)."""
        # Qdrant upsert handles both insert and update
        return self.add_documents(documents, collection_name)
    
    async def async_update_documents(
        self,
        documents: Sequence[DocumentChunk],
        collection_name: Optional[str] = None
    ) -> List[str]:
        """Async version of update_documents."""
        return await self.async_add_documents(documents, collection_name)
    
    def delete_by_ids(
        self,
        ids: List[str],
        collection_name: Optional[str] = None
    ) -> bool:
        """Delete documents by ID from Qdrant."""
        name = self._get_collection_name(collection_name)
        
        result = self._client.delete(
            collection_name=name,
            points_selector=PointIdsList(points=ids),
            wait=True
        )
        
        return result.status == UpdateStatus.COMPLETED
    
    async def async_delete_by_ids(
        self,
        ids: List[str],
        collection_name: Optional[str] = None
    ) -> bool:
        """Async version of delete_by_ids."""
        name = self._get_collection_name(collection_name)
        
        result = await self._async_client.delete(
            collection_name=name,
            points_selector=PointIdsList(points=ids),
            wait=True
        )
        
        return result.status == UpdateStatus.COMPLETED
    
    def delete_by_filter(
        self,
        filters: Union[MetadataFilter, MetadataFilterGroup],
        collection_name: Optional[str] = None
    ) -> int:
        """Delete documents matching filter from Qdrant."""
        name = self._get_collection_name(collection_name)
        qdrant_filter = self._build_qdrant_filter(filters)
        
        # Count before delete
        count_before = self.count(collection_name)
        
        self._client.delete(
            collection_name=name,
            points_selector=qdrant_models.FilterSelector(filter=qdrant_filter),
            wait=True
        )
        
        # Count after delete
        count_after = self.count(collection_name)
        
        return count_before - count_after
    
    async def async_delete_by_filter(
        self,
        filters: Union[MetadataFilter, MetadataFilterGroup],
        collection_name: Optional[str] = None
    ) -> int:
        """Async version of delete_by_filter."""
        name = self._get_collection_name(collection_name)
        qdrant_filter = self._build_qdrant_filter(filters)
        
        count_before = await self.async_count(collection_name)
        
        await self._async_client.delete(
            collection_name=name,
            points_selector=qdrant_models.FilterSelector(filter=qdrant_filter),
            wait=True
        )
        
        count_after = await self.async_count(collection_name)
        
        return count_before - count_after
    
    # ==================== Search Operations ====================
    
    def search(
        self,
        query_embedding: List[float],
        k: int = 10,
        filters: Optional[Union[MetadataFilter, MetadataFilterGroup]] = None,
        collection_name: Optional[str] = None,
        score_threshold: Optional[float] = None
    ) -> List[SearchResult]:
        """Search for similar documents in Qdrant."""
        name = self._get_collection_name(collection_name)
        qdrant_filter = self._build_qdrant_filter(filters)
        
        response = self._client.query_points(
            collection_name=name,
            query=query_embedding,
            limit=k,
            query_filter=qdrant_filter,
            score_threshold=score_threshold,
            with_payload=True,
            with_vectors=False
        )
        
        search_results = []
        for point in response.points:
            payload = point.payload or {}
            content = payload.pop("content", "")
            
            search_results.append(SearchResult(
                id=str(point.id),
                content=content,
                score=point.score,
                metadata=payload
            ))
        
        return search_results
    
    async def async_search(
        self,
        query_embedding: List[float],
        k: int = 10,
        filters: Optional[Union[MetadataFilter, MetadataFilterGroup]] = None,
        collection_name: Optional[str] = None,
        score_threshold: Optional[float] = None
    ) -> List[SearchResult]:
        """Async version of search."""
        name = self._get_collection_name(collection_name)
        qdrant_filter = self._build_qdrant_filter(filters)
        
        response = await self._async_client.query_points(
            collection_name=name,
            query=query_embedding,
            limit=k,
            query_filter=qdrant_filter,
            score_threshold=score_threshold,
            with_payload=True,
            with_vectors=False
        )
        
        search_results = []
        for point in response.points:
            payload = point.payload or {}
            content = payload.pop("content", "")
            
            search_results.append(SearchResult(
                id=str(point.id),
                content=content,
                score=point.score,
                metadata=payload
            ))
        
        return search_results
    
    def similarity_search_with_score(
        self,
        query_embedding: List[float],
        k: int = 10,
        filters: Optional[Union[MetadataFilter, MetadataFilterGroup]] = None,
        collection_name: Optional[str] = None
    ) -> List[tuple[DocumentChunk, float]]:
        """Search and return documents with scores."""
        name = self._get_collection_name(collection_name)
        qdrant_filter = self._build_qdrant_filter(filters)
        
        response = self._client.query_points(
            collection_name=name,
            query=query_embedding,
            limit=k,
            query_filter=qdrant_filter,
            with_payload=True,
            with_vectors=True
        )
        
        doc_scores = []
        for point in response.points:
            payload = point.payload or {}
            content = payload.pop("content", "")
            
            doc = DocumentChunk(
                id=str(point.id),
                content=content,
                metadata=payload,
                embedding=point.vector if point.vector else None
            )
            doc_scores.append((doc, point.score))
        
        return doc_scores
    
    async def async_similarity_search_with_score(
        self,
        query_embedding: List[float],
        k: int = 10,
        filters: Optional[Union[MetadataFilter, MetadataFilterGroup]] = None,
        collection_name: Optional[str] = None
    ) -> List[tuple[DocumentChunk, float]]:
        """Async version of similarity_search_with_score."""
        name = self._get_collection_name(collection_name)
        qdrant_filter = self._build_qdrant_filter(filters)
        
        response = await self._async_client.query_points(
            collection_name=name,
            query=query_embedding,
            limit=k,
            query_filter=qdrant_filter,
            with_payload=True,
            with_vectors=True
        )
        
        doc_scores = []
        for point in response.points:
            payload = point.payload or {}
            content = payload.pop("content", "")
            
            doc = DocumentChunk(
                id=str(point.id),
                content=content,
                metadata=payload,
                embedding=point.vector if point.vector else None
            )
            doc_scores.append((doc, point.score))
        
        return doc_scores
    
    # ==================== Utility Operations ====================
    
    def count(self, collection_name: Optional[str] = None) -> int:
        """Count documents in Qdrant collection."""
        name = self._get_collection_name(collection_name)
        info = self._client.get_collection(collection_name=name)
        return info.points_count or 0
    
    async def async_count(self, collection_name: Optional[str] = None) -> int:
        """Async version of count."""
        name = self._get_collection_name(collection_name)
        info = await self._async_client.get_collection(collection_name=name)
        return info.points_count or 0
