"""
Asset Upload Service

Service for uploading files and directories to vector databases.
Designed for personal asset management and batch document ingestion.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from RAGService.Data.VectorDB import (
    BaseVectorDB,
    DocumentChunk,
    VectorDBConfig,
    VectorDBFactory,
    VectorDBProvider,
    DistanceMetric,
)
from RAGService.Data.Embeddings import (
    BaseEmbeddings,
    EmbeddingConfig,
    EmbeddingsFactory,
    EmbeddingInputType,
    EmbeddingProvider,
)
from RAGService.Data.DocumentProcessors import (
    DocumentLoaderFactory,
    ProcessedDocument,
    RecursiveTextSplitter,
    TextSplitterConfig,
)
from RAGService.Data.DocumentProcessors.smart_chunker import (
    SmartChunker,
    SmartChunkerConfig,
)


@dataclass
class UploadResult:
    """
    Result of an upload operation.
    
    Attributes:
        success: Whether the upload was successful
        document_ids: List of IDs of uploaded documents
        total_chunks: Total number of chunks created
        source: Source file or directory path
        error: Error message if upload failed
        metadata: Additional metadata about the upload
    """
    success: bool
    document_ids: List[str] = field(default_factory=list)
    total_chunks: int = 0
    source: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AssetUploadConfig:
    """
    Configuration for AssetUploadService.
    
    Attributes:
        vectordb_provider: Vector database provider
        embedding_provider: Embedding model provider
        vectordb_url: URL for vector database
        vectordb_api_key: API key for vector database
        embedding_api_key: API key for embedding model
        embedding_model: Embedding model name
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        batch_size: Batch size for embedding and insertion
        default_collection: Default collection name
    """
    vectordb_provider: VectorDBProvider = VectorDBProvider.QDRANT
    embedding_provider: EmbeddingProvider = EmbeddingProvider.COHERE
    vectordb_url: Optional[str] = None
    vectordb_api_key: Optional[str] = None
    embedding_api_key: Optional[str] = None
    embedding_model: Optional[str] = None
    chunk_size: int = 1000
    chunk_overlap: int = 200
    batch_size: int = 96
    default_collection: str = "assets"
    distance_metric: DistanceMetric = DistanceMetric.COSINE
    
    # Smart chunking
    use_smart_chunker: bool = True
    use_llm_analysis: bool = False
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> "AssetUploadConfig":
        """Create configuration from environment variables."""
        return cls(
            vectordb_provider=VectorDBProvider(
                os.environ.get("VECTORDB_PROVIDER", "qdrant")
            ),
            embedding_provider=EmbeddingProvider(
                os.environ.get("EMBEDDING_PROVIDER", "cohere")
            ),
            vectordb_url=os.environ.get("QDRANT_URL"),
            vectordb_api_key=os.environ.get("QDRANT_API_KEY"),
            embedding_api_key=os.environ.get("COHERE_API_KEY"),
            embedding_model=os.environ.get("EMBEDDING_MODEL"),
            chunk_size=int(os.environ.get("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.environ.get("CHUNK_OVERLAP", "200")),
            default_collection=os.environ.get("DEFAULT_COLLECTION", "assets"),
        )


class AssetUploadService:
    """
    Service for uploading assets to vector databases.
    
    Provides methods for uploading files, directories, and raw text
    to a vector database with automatic chunking and embedding.
    
    This service is designed for personal asset management, allowing
    you to build up a knowledge base from various document formats.
    
    Example:
        # Create service
        service = AssetUploadService(
            config=AssetUploadConfig(
                vectordb_url="http://localhost:6333",
                default_collection="my_docs"
            )
        )
        
        # Upload a file
        result = service.upload_file("document.pdf", collection_name="pdfs")
        
        # Upload a directory
        result = service.upload_directory("./docs", recursive=True)
        
        # Upload raw text
        result = service.upload_text("Hello world", metadata={"source": "manual"})
    """
    
    def __init__(
        self,
        config: Optional[AssetUploadConfig] = None,
        vectordb: Optional[BaseVectorDB] = None,
        embeddings: Optional[BaseEmbeddings] = None
    ):
        """
        Initialize the asset upload service.
        
        Args:
            config: Service configuration
            vectordb: Pre-configured VectorDB instance
            embeddings: Pre-configured embeddings instance
        """
        self.config = config or AssetUploadConfig.from_env()
        
        # Initialize embeddings
        if embeddings:
            self._embeddings = embeddings
        else:
            self._embeddings = self._create_embeddings()
        
        # Initialize vector database
        if vectordb:
            self._vectordb = vectordb
        else:
            self._vectordb = self._create_vectordb()
        
        # Initialize chunking
        if self.config.use_smart_chunker:
            smart_config = SmartChunkerConfig(
                max_chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                use_llm_analysis=self.config.use_llm_analysis,
                llm_provider=self.config.llm_provider,
                llm_model=self.config.llm_model,
            )
            self._smart_chunker = SmartChunker(smart_config)
        else:
            self._smart_chunker = None
        
        # Fallback splitter for raw text uploads
        self._splitter = RecursiveTextSplitter(
            TextSplitterConfig(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
        )
        
        # Track created collections
        self._created_collections: set = set()
    
    def _create_embeddings(self) -> BaseEmbeddings:
        """Create embeddings instance from config."""
        if self.config.embedding_provider == EmbeddingProvider.COHERE:
            return EmbeddingsFactory.create_cohere(
                api_key=self.config.embedding_api_key,
                model_name=self.config.embedding_model or "embed-english-v3.0"
            )
        elif self.config.embedding_provider == EmbeddingProvider.OPENAI:
            return EmbeddingsFactory.create_openai(
                api_key=self.config.embedding_api_key,
                model_name=self.config.embedding_model or "text-embedding-3-small"
            )
        else:
            return EmbeddingsFactory.create_from_env(self.config.embedding_provider)
    
    def _create_vectordb(self) -> BaseVectorDB:
        """Create VectorDB instance from config."""
        if self.config.vectordb_provider == VectorDBProvider.QDRANT:
            return VectorDBFactory.create_qdrant(
                collection_name=self.config.default_collection,
                embedding_dimension=self._embeddings.dimension,
                url=self.config.vectordb_url,
                api_key=self.config.vectordb_api_key,
                distance_metric=self.config.distance_metric,
                in_memory=(self.config.vectordb_url is None)
            )
        else:
            vectordb_config = VectorDBConfig(
                provider=self.config.vectordb_provider,
                collection_name=self.config.default_collection,
                embedding_dimension=self._embeddings.dimension,
                url=self.config.vectordb_url,
                api_key=self.config.vectordb_api_key,
                distance_metric=self.config.distance_metric,
            )
            return VectorDBFactory.create(vectordb_config)
    
    def _ensure_collection(self, collection_name: str) -> None:
        """Ensure a collection exists, creating if necessary."""
        if collection_name not in self._created_collections:
            if not self._vectordb.collection_exists(collection_name):
                self._vectordb.create_collection(
                    collection_name=collection_name,
                    dimension=self._embeddings.dimension
                )
            self._created_collections.add(collection_name)
    
    async def _async_ensure_collection(self, collection_name: str) -> None:
        """Async version of _ensure_collection."""
        if collection_name not in self._created_collections:
            if not await self._vectordb.async_collection_exists(collection_name):
                await self._vectordb.async_create_collection(
                    collection_name=collection_name,
                    dimension=self._embeddings.dimension
                )
            self._created_collections.add(collection_name)
    
    def _embed_chunks(
        self,
        chunks: List[DocumentChunk],
    ) -> None:
        """Generate embeddings for chunks that don't have them."""
        texts = [chunk.content for chunk in chunks]
        embeddings = self._embeddings.embed_documents(
            texts,
            input_type=EmbeddingInputType.SEARCH_DOCUMENT,
            batch_size=self.config.batch_size
        )
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
    
    async def _async_embed_chunks(
        self,
        chunks: List[DocumentChunk],
    ) -> None:
        """Async version: generate embeddings for chunks."""
        texts = [chunk.content for chunk in chunks]
        embeddings = await self._embeddings.async_embed_documents(
            texts,
            input_type=EmbeddingInputType.SEARCH_DOCUMENT,
            batch_size=self.config.batch_size
        )
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
    
    # ==================== File Upload ====================
    
    def upload_file(
        self,
        file_path: Union[str, Path],
        collection_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **loader_kwargs
    ) -> UploadResult:
        """
        Upload a single file to the vector database.
        
        Args:
            file_path: Path to the file
            collection_name: Target collection (uses default if not provided)
            metadata: Additional metadata to attach to all chunks
            **loader_kwargs: Additional arguments for the document loader
            
        Returns:
            UploadResult with status and details
        """
        try:
            path = Path(file_path)
            collection = collection_name or self.config.default_collection
            
            # Ensure collection exists
            self._ensure_collection(collection)
            
            # Load + chunk using SmartChunker or fallback
            if self._smart_chunker:
                chunks = self._smart_chunker.chunk_file(
                    path, metadata=metadata, **loader_kwargs
                )
                file_type = path.suffix.lower().lstrip(".")
            else:
                doc = DocumentLoaderFactory.load(path, metadata, **loader_kwargs)
                chunks = self._splitter.split_document(doc)
                file_type = doc.file_type.value if doc.file_type else None
            
            if not chunks:
                return UploadResult(
                    success=True,
                    total_chunks=0,
                    source=str(path),
                    metadata={"message": "No content to upload"}
                )
            
            # Generate embeddings
            self._embed_chunks(chunks)
            
            # Upload to database
            ids = self._vectordb.add_documents(
                chunks,
                collection_name=collection,
                batch_size=self.config.batch_size
            )
            
            return UploadResult(
                success=True,
                document_ids=ids,
                total_chunks=len(chunks),
                source=str(path),
                metadata={
                    "file_type": file_type,
                    "collection": collection
                }
            )
        
        except Exception as e:
            return UploadResult(
                success=False,
                source=str(file_path),
                error=str(e)
            )
    
    async def async_upload_file(
        self,
        file_path: Union[str, Path],
        collection_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **loader_kwargs
    ) -> UploadResult:
        """Async version of upload_file."""
        try:
            path = Path(file_path)
            collection = collection_name or self.config.default_collection
            
            await self._async_ensure_collection(collection)
            
            # Load + chunk using SmartChunker or fallback
            if self._smart_chunker:
                chunks = await self._smart_chunker.async_chunk_file(
                    path, metadata=metadata, **loader_kwargs
                )
                file_type = path.suffix.lower().lstrip(".")
            else:
                doc = await DocumentLoaderFactory.async_load(path, metadata, **loader_kwargs)
                chunks = self._splitter.split_document(doc)
                file_type = doc.file_type.value if doc.file_type else None
            
            if not chunks:
                return UploadResult(
                    success=True,
                    total_chunks=0,
                    source=str(path),
                    metadata={"message": "No content to upload"}
                )
            
            await self._async_embed_chunks(chunks)
            
            ids = await self._vectordb.async_add_documents(
                chunks,
                collection_name=collection,
                batch_size=self.config.batch_size
            )
            
            return UploadResult(
                success=True,
                document_ids=ids,
                total_chunks=len(chunks),
                source=str(path),
                metadata={
                    "file_type": file_type,
                    "collection": collection
                }
            )
        
        except Exception as e:
            return UploadResult(
                success=False,
                source=str(file_path),
                error=str(e)
            )
    
    # ==================== Directory Upload ====================
    
    def upload_directory(
        self,
        directory_path: Union[str, Path],
        collection_name: Optional[str] = None,
        recursive: bool = True,
        extensions: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **loader_kwargs
    ) -> List[UploadResult]:
        """
        Upload all supported files from a directory.
        
        Args:
            directory_path: Path to the directory
            collection_name: Target collection
            recursive: Whether to process subdirectories
            extensions: List of extensions to process (None = all supported)
            metadata: Additional metadata for all files
            **loader_kwargs: Additional arguments for loaders
            
        Returns:
            List of UploadResult for each file
        """
        results = []
        path = Path(directory_path)
        
        # Get supported extensions
        supported = set(DocumentLoaderFactory.list_supported_extensions())
        if extensions:
            filter_exts = set(ext.lower().lstrip(".") for ext in extensions)
            supported = supported & filter_exts
        
        # Find files
        pattern = "**/*" if recursive else "*"
        files = [
            f for f in path.glob(pattern)
            if f.is_file() and f.suffix.lower().lstrip(".") in supported
        ]
        
        # Upload each file
        for file_path in files:
            # Merge directory metadata with file-specific metadata
            file_metadata = {**(metadata or {}), "directory": str(path)}
            result = self.upload_file(
                file_path,
                collection_name=collection_name,
                metadata=file_metadata,
                **loader_kwargs
            )
            results.append(result)
        
        return results
    
    async def async_upload_directory(
        self,
        directory_path: Union[str, Path],
        collection_name: Optional[str] = None,
        recursive: bool = True,
        extensions: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **loader_kwargs
    ) -> List[UploadResult]:
        """Async version of upload_directory."""
        import asyncio
        
        results = []
        path = Path(directory_path)
        
        supported = set(DocumentLoaderFactory.list_supported_extensions())
        if extensions:
            filter_exts = set(ext.lower().lstrip(".") for ext in extensions)
            supported = supported & filter_exts
        
        pattern = "**/*" if recursive else "*"
        files = [
            f for f in path.glob(pattern)
            if f.is_file() and f.suffix.lower().lstrip(".") in supported
        ]
        
        # Process files concurrently (with limit to avoid overwhelming API)
        semaphore = asyncio.Semaphore(5)
        
        async def upload_with_limit(file_path: Path) -> UploadResult:
            async with semaphore:
                file_metadata = {**(metadata or {}), "directory": str(path)}
                return await self.async_upload_file(
                    file_path,
                    collection_name=collection_name,
                    metadata=file_metadata,
                    **loader_kwargs
                )
        
        results = await asyncio.gather(*[upload_with_limit(f) for f in files])
        return list(results)
    
    # ==================== Text Upload ====================
    
    def upload_text(
        self,
        text: str,
        collection_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None
    ) -> UploadResult:
        """
        Upload raw text to the vector database.
        
        Args:
            text: Text content to upload
            collection_name: Target collection
            metadata: Additional metadata
            source: Source identifier for the text
            
        Returns:
            UploadResult with status and details
        """
        try:
            collection = collection_name or self.config.default_collection
            self._ensure_collection(collection)
            
            # Create a processed document
            doc = ProcessedDocument(
                content=text,
                metadata=metadata or {},
                source=source or "manual_input"
            )
            
            # Split into chunks
            chunks = self._splitter.split_document(doc)
            
            if not chunks:
                return UploadResult(
                    success=True,
                    total_chunks=0,
                    source=source,
                    metadata={"message": "No content to upload"}
                )
            
            # Generate embeddings
            self._embed_chunks(chunks)
            
            # Upload
            ids = self._vectordb.add_documents(
                chunks,
                collection_name=collection,
                batch_size=self.config.batch_size
            )
            
            return UploadResult(
                success=True,
                document_ids=ids,
                total_chunks=len(chunks),
                source=source,
                metadata={"collection": collection}
            )
        
        except Exception as e:
            return UploadResult(
                success=False,
                source=source,
                error=str(e)
            )
    
    async def async_upload_text(
        self,
        text: str,
        collection_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None
    ) -> UploadResult:
        """Async version of upload_text."""
        try:
            collection = collection_name or self.config.default_collection
            await self._async_ensure_collection(collection)
            
            doc = ProcessedDocument(
                content=text,
                metadata=metadata or {},
                source=source or "manual_input"
            )
            
            chunks = self._splitter.split_document(doc)
            
            if not chunks:
                return UploadResult(
                    success=True,
                    total_chunks=0,
                    source=source,
                    metadata={"message": "No content to upload"}
                )
            
            await self._async_embed_chunks(chunks)
            
            ids = await self._vectordb.async_add_documents(
                chunks,
                collection_name=collection,
                batch_size=self.config.batch_size
            )
            
            return UploadResult(
                success=True,
                document_ids=ids,
                total_chunks=len(chunks),
                source=source,
                metadata={"collection": collection}
            )
        
        except Exception as e:
            return UploadResult(
                success=False,
                source=source,
                error=str(e)
            )
    
    # ==================== Batch Upload ====================
    
    def upload_texts(
        self,
        texts: List[str],
        collection_name: Optional[str] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        sources: Optional[List[str]] = None
    ) -> UploadResult:
        """
        Upload multiple text strings as separate documents.
        
        Each text is treated as a complete document and will be
        chunked independently.
        
        Args:
            texts: List of text strings
            collection_name: Target collection
            metadatas: Optional metadata for each text
            sources: Optional source identifiers
            
        Returns:
            UploadResult with combined stats
        """
        try:
            collection = collection_name or self.config.default_collection
            self._ensure_collection(collection)
            
            all_chunks = []
            
            for i, text in enumerate(texts):
                metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
                source = sources[i] if sources and i < len(sources) else f"text_{i}"
                
                doc = ProcessedDocument(
                    content=text,
                    metadata=metadata,
                    source=source
                )
                
                chunks = self._splitter.split_document(doc)
                all_chunks.extend(chunks)
            
            if not all_chunks:
                return UploadResult(
                    success=True,
                    total_chunks=0,
                    metadata={"message": "No content to upload"}
                )
            
            # Generate embeddings for all chunks
            self._embed_chunks(all_chunks)
            
            ids = self._vectordb.add_documents(
                all_chunks,
                collection_name=collection,
                batch_size=self.config.batch_size
            )
            
            return UploadResult(
                success=True,
                document_ids=ids,
                total_chunks=len(all_chunks),
                metadata={
                    "collection": collection,
                    "document_count": len(texts)
                }
            )
        
        except Exception as e:
            return UploadResult(
                success=False,
                error=str(e)
            )
    
    async def async_upload_texts(
        self,
        texts: List[str],
        collection_name: Optional[str] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        sources: Optional[List[str]] = None
    ) -> UploadResult:
        """Async version of upload_texts."""
        try:
            collection = collection_name or self.config.default_collection
            await self._async_ensure_collection(collection)
            
            all_chunks = []
            
            for i, text in enumerate(texts):
                metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
                source = sources[i] if sources and i < len(sources) else f"text_{i}"
                
                doc = ProcessedDocument(
                    content=text,
                    metadata=metadata,
                    source=source
                )
                
                chunks = self._splitter.split_document(doc)
                all_chunks.extend(chunks)
            
            if not all_chunks:
                return UploadResult(
                    success=True,
                    total_chunks=0,
                    metadata={"message": "No content to upload"}
                )
            
            await self._async_embed_chunks(all_chunks)
            
            ids = await self._vectordb.async_add_documents(
                all_chunks,
                collection_name=collection,
                batch_size=self.config.batch_size
            )
            
            return UploadResult(
                success=True,
                document_ids=ids,
                total_chunks=len(all_chunks),
                metadata={
                    "collection": collection,
                    "document_count": len(texts)
                }
            )
        
        except Exception as e:
            return UploadResult(
                success=False,
                error=str(e)
            )
    
    # ==================== Utility Methods ====================
    
    def list_collections(self) -> List[str]:
        """List all collections in the database."""
        return self._vectordb.list_collections()
    
    async def async_list_collections(self) -> List[str]:
        """Async version of list_collections."""
        return await self._vectordb.async_list_collections()
    
    def get_collection_stats(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics for a collection."""
        collection = collection_name or self.config.default_collection
        
        if not self._vectordb.collection_exists(collection):
            return {"exists": False, "collection": collection}
        
        info = self._vectordb.get_collection_info(collection)
        return {
            "exists": True,
            "collection": collection,
            "vector_count": info.vector_count,
            "dimension": info.dimension,
            "distance_metric": info.distance_metric.value,
            **info.metadata
        }
    
    async def async_get_collection_stats(
        self,
        collection_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Async version of get_collection_stats."""
        collection = collection_name or self.config.default_collection
        
        if not await self._vectordb.async_collection_exists(collection):
            return {"exists": False, "collection": collection}
        
        info = await self._vectordb.async_get_collection_info(collection)
        return {
            "exists": True,
            "collection": collection,
            "vector_count": info.vector_count,
            "dimension": info.dimension,
            "distance_metric": info.distance_metric.value,
            **info.metadata
        }
    
    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection."""
        result = self._vectordb.delete_collection(collection_name)
        self._created_collections.discard(collection_name)
        return result
    
    async def async_delete_collection(self, collection_name: str) -> bool:
        """Async version of delete_collection."""
        result = await self._vectordb.async_delete_collection(collection_name)
        self._created_collections.discard(collection_name)
        return result


# Global singleton instance
_asset_upload_service_instance: Optional[AssetUploadService] = None


def get_asset_upload_service(
    config: Optional[AssetUploadConfig] = None,
    reset: bool = False
) -> AssetUploadService:
    """
    Get the global AssetUploadService singleton.
    
    Args:
        config: Service configuration (only used on first call or reset)
        reset: Whether to reset the singleton
        
    Returns:
        AssetUploadService instance
    """
    global _asset_upload_service_instance
    
    if _asset_upload_service_instance is None or reset:
        _asset_upload_service_instance = AssetUploadService(config)
    
    return _asset_upload_service_instance
