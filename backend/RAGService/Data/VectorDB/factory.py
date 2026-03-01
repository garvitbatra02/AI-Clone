"""
VectorDB Factory

Factory class for creating vector database instances.
Follows the same pattern as the LLM factory in ChatService.
"""

from typing import Any, Dict, Optional, Type

from RAGService.Data.VectorDB.base import (
    BaseVectorDB,
    DistanceMetric,
    VectorDBConfig,
    VectorDBProvider,
)
from RAGService.Data.VectorDB.registry import (
    get_provider_class,
    is_provider_available,
    list_available_providers,
    register_provider,
)


class VectorDBFactory:
    """
    Factory for creating vector database instances.
    
    Provides methods to create VectorDB instances from configuration
    objects or environment variables.
    
    Example:
        # Create from config
        config = VectorDBConfig.for_qdrant(
            collection_name="my_collection",
            embedding_dimension=768,
        )
        db = VectorDBFactory.create(config)
        
        # Create from environment
        db = VectorDBFactory.create_from_env(
            provider=VectorDBProvider.QDRANT,
            collection_name="my_collection",
            embedding_dimension=768
        )
    """
    
    @staticmethod
    def create(config: VectorDBConfig) -> BaseVectorDB:
        """
        Create a vector database instance from configuration.
        
        Args:
            config: VectorDBConfig with connection details
            
        Returns:
            Configured BaseVectorDB instance
            
        Raises:
            ValueError: If the provider is not available
        """
        provider_class = get_provider_class(config.provider)
        return provider_class(config)
    
    @staticmethod
    def create_from_env(
        provider: VectorDBProvider,
        collection_name: str,
        embedding_dimension: int,
        distance_metric: DistanceMetric = DistanceMetric.COSINE,
        **extra_config
    ) -> BaseVectorDB:
        """
        Create a vector database instance from environment variables.
        
        The provider implementation resolves its own API key and URL
        from environment variables when not explicitly provided in config.
        
        Args:
            provider: The vector database provider
            collection_name: Name of the collection
            embedding_dimension: Dimension of embedding vectors
            distance_metric: Distance metric for similarity
            **extra_config: Additional provider-specific configuration
            
        Returns:
            Configured BaseVectorDB instance
        """
        config = VectorDBConfig(
            provider=provider,
            collection_name=collection_name,
            embedding_dimension=embedding_dimension,
            distance_metric=distance_metric,
            extra_config=extra_config
        )
        
        return VectorDBFactory.create(config)
    
    @staticmethod
    def create_qdrant(
        collection_name: str,
        embedding_dimension: int,
        distance_metric: DistanceMetric = DistanceMetric.COSINE,
        in_memory: bool = False,
        path: Optional[str] = None,
        **extra_config
    ) -> BaseVectorDB:
        """
        Convenience method to create a Qdrant instance.
        
        The Qdrant provider resolves its own URL and API key from
        environment variables (QDRANT_URL, QDRANT_API_KEY) when not
        explicitly provided in config.
        
        Args:
            collection_name: Name of the collection
            embedding_dimension: Dimension of embedding vectors
            distance_metric: Distance metric for similarity
            in_memory: Use in-memory storage (for testing)
            path: Local file path for persistent storage
            **extra_config: Additional configuration
            
        Returns:
            Configured QdrantVectorDB instance
        """
        extra = {
            "in_memory": in_memory,
            "path": path,
            **extra_config
        }
        
        config = VectorDBConfig(
            provider=VectorDBProvider.QDRANT,
            collection_name=collection_name,
            embedding_dimension=embedding_dimension,
            distance_metric=distance_metric,
            extra_config=extra
        )
        
        return VectorDBFactory.create(config)
    
    @staticmethod
    def register_provider(
        provider: VectorDBProvider,
        implementation: Type[BaseVectorDB],
        override: bool = False
    ) -> None:
        """
        Register a custom provider implementation.
        
        Args:
            provider: The provider enum value
            implementation: The implementation class
            override: Whether to override existing registration
        """
        register_provider(provider, implementation, override)
    
    @staticmethod
    def list_providers() -> list[VectorDBProvider]:
        """
        List all available providers.
        
        Returns:
            List of available VectorDBProvider enums
        """
        return list_available_providers()
    
    @staticmethod
    def is_available(provider: VectorDBProvider) -> bool:
        """
        Check if a provider is available.
        
        Args:
            provider: The provider to check
            
        Returns:
            True if the provider is registered and available
        """
        return is_provider_available(provider)
