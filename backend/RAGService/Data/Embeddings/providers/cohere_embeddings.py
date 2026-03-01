"""
Cohere Embeddings Implementation

Provides embedding functionality using Cohere's embedding models.
"""

from __future__ import annotations

import asyncio
import os
from typing import List, Optional

import cohere
from cohere import AsyncClientV2, ClientV2

from RAGService.Data.Embeddings.base import (
    BaseEmbeddings,
    EmbeddingConfig,
    EmbeddingInputType,
    EmbeddingProvider,
)


class CohereEmbeddings(BaseEmbeddings):
    """
    Cohere implementation of the Embeddings interface.
    
    Supports Cohere's embed-v3 models with various configurations:
    - embed-english-v3.0: 1024 dimensions, English text
    - embed-english-light-v3.0: 384 dimensions, English text (faster)
    - embed-multilingual-v3.0: 1024 dimensions, 100+ languages
    - embed-multilingual-light-v3.0: 384 dimensions, 100+ languages (faster)
    
    Cohere embeddings support input types that optimize for different use cases:
    - search_document: For documents to be indexed and searched
    - search_query: For search queries
    - classification: For classification tasks
    - clustering: For clustering tasks
    
    Example:
        config = EmbeddingConfig.for_cohere(
            model_name="embed-english-v3.0"
        )
        embeddings = CohereEmbeddings(config)
        
        # Embed a query
        query_embedding = embeddings.embed_query("What is machine learning?")
        
        # Embed documents
        doc_embeddings = embeddings.embed_documents([
            "Machine learning is a type of AI.",
            "Deep learning uses neural networks."
        ])
    """
    
    ENV_VAR_NAME: str = "COHERE_API_KEY"
    
    _client: ClientV2
    _async_client: AsyncClientV2
    _dimension: int
    
    # Cohere input type mapping
    _INPUT_TYPE_MAP = {
        EmbeddingInputType.SEARCH_DOCUMENT: "search_document",
        EmbeddingInputType.SEARCH_QUERY: "search_query",
        EmbeddingInputType.CLASSIFICATION: "classification",
        EmbeddingInputType.CLUSTERING: "clustering",
    }
    
    # Model dimension mapping
    _DIMENSION_MAP = {
        "embed-english-v3.0": 1024,
        "embed-english-light-v3.0": 384,
        "embed-multilingual-v3.0": 1024,
        "embed-multilingual-light-v3.0": 384,
        # Legacy models
        "embed-english-v2.0": 4096,
        "embed-english-light-v2.0": 1024,
        "embed-multilingual-v2.0": 768,
    }
    
    def _initialize_client(self) -> None:
        """Initialize Cohere clients (sync and async)."""
        api_key = self.config.api_key or os.environ.get(self.ENV_VAR_NAME)
        
        if not api_key:
            raise ValueError(
                f"Cohere API key is required. Provide it via config or "
                f"set the {self.ENV_VAR_NAME} environment variable."
            )
        
        self._client = ClientV2(
            api_key=api_key,
            timeout=self.config.timeout
        )
        self._async_client = AsyncClientV2(
            api_key=api_key,
            timeout=self.config.timeout
        )
        
        # Set dimension based on model
        self._dimension = (
            self.config.dimension or
            self._DIMENSION_MAP.get(self.config.model_name, 1024)
        )
    
    @property
    def provider(self) -> EmbeddingProvider:
        return EmbeddingProvider.COHERE
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    def _get_input_type(
        self,
        input_type: Optional[EmbeddingInputType],
        default: str = "search_query"
    ) -> str:
        """Convert EmbeddingInputType to Cohere input type string."""
        if input_type is None:
            return default
        return self._INPUT_TYPE_MAP.get(input_type, default)
    
    def embed_query(
        self,
        text: str,
        input_type: Optional[EmbeddingInputType] = None
    ) -> List[float]:
        """
        Embed a single query text.
        
        Args:
            text: The text to embed
            input_type: Input type hint (defaults to search_query)
            
        Returns:
            Embedding vector as list of floats
        """
        cohere_input_type = self._get_input_type(
            input_type, default="search_query"
        )
        
        response = self._client.embed(
            texts=[text],
            model=self.config.model_name,
            input_type=cohere_input_type,
            embedding_types=["float"],
            truncate=self.config.truncate
        )
        
        # Extract embedding from response
        # V2 API returns embeddings in response.embeddings.float_
        return response.embeddings.float_[0]
    
    async def async_embed_query(
        self,
        text: str,
        input_type: Optional[EmbeddingInputType] = None
    ) -> List[float]:
        """Async version of embed_query."""
        cohere_input_type = self._get_input_type(
            input_type, default="search_query"
        )
        
        response = await self._async_client.embed(
            texts=[text],
            model=self.config.model_name,
            input_type=cohere_input_type,
            embedding_types=["float"],
            truncate=self.config.truncate
        )
        
        return response.embeddings.float_[0]
    
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
            input_type: Input type hint (defaults to search_document)
            batch_size: Override default batch size (max 96 for Cohere)
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        cohere_input_type = self._get_input_type(
            input_type, default="search_document"
        )
        
        batch = batch_size or self.config.batch_size
        # Cohere max batch size is 96
        batch = min(batch, 96)
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch):
            batch_texts = texts[i:i + batch]
            
            response = self._client.embed(
                texts=batch_texts,
                model=self.config.model_name,
                input_type=cohere_input_type,
                embedding_types=["float"],
                truncate=self.config.truncate
            )
            
            all_embeddings.extend(response.embeddings.float_)
        
        return all_embeddings
    
    async def async_embed_documents(
        self,
        texts: List[str],
        input_type: Optional[EmbeddingInputType] = None,
        batch_size: Optional[int] = None
    ) -> List[List[float]]:
        """Async version of embed_documents."""
        if not texts:
            return []
        
        cohere_input_type = self._get_input_type(
            input_type, default="search_document"
        )
        
        batch = batch_size or self.config.batch_size
        batch = min(batch, 96)
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch):
            batch_texts = texts[i:i + batch]
            
            response = await self._async_client.embed(
                texts=batch_texts,
                model=self.config.model_name,
                input_type=cohere_input_type,
                embedding_types=["float"],
                truncate=self.config.truncate
            )
            
            all_embeddings.extend(response.embeddings.float_)
        
        return all_embeddings
