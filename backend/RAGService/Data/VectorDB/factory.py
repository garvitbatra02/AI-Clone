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
        # Create from config object
        config = VectorDBConfig.for_qdrant(
            collection_name="my_collection",
            embedding_dimension=768,
        )
        db = VectorDBFactory.create(config)
        
        # Create with inline params (provider resolves secrets from env)
        db = VectorDBFactory.create_from_env(
            provider=VectorDBProvider.QDRANT,
            collection_name="my_collection",
            embedding_dimension=768,
            in_memory=True,
        )
    """
    
    @staticmethod
    def create(config: VectorDBConfig) -> BaseVectorDB:
        """
        Create a vector database instance from a pre-built configuration.
        
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
        timeout: Optional[int] = None,
        prefer_grpc: Optional[bool] = None,
        **extra_config
    ) -> BaseVectorDB:
        """
        Create a vector database instance with inline parameters.
        
        Builds a VectorDBConfig and delegates to create(). The provider
        implementation resolves its own API key and URL from environment
        variables.
        
        Provider-specific options (e.g. in_memory, path for Qdrant) are
        passed through **extra_config.
        
        Args:
            provider: The vector database provider
            collection_name: Name of the collection
            embedding_dimension: Dimension of embedding vectors
            distance_metric: Distance metric for similarity
            timeout: Connection timeout in seconds (default: 30)
            prefer_grpc: Whether to prefer gRPC over HTTP (default: True)
            **extra_config: Provider-specific config (e.g. in_memory, path)
            
        Returns:
            Configured BaseVectorDB instance
        """
        # Build optional kwargs so VectorDBConfig defaults are preserved
        # when the caller doesn't explicitly override them.
        optional: Dict[str, Any] = {}
        if timeout is not None:
            optional["timeout"] = timeout
        if prefer_grpc is not None:
            optional["prefer_grpc"] = prefer_grpc
        
        config = VectorDBConfig(
            provider=provider,
            collection_name=collection_name,
            embedding_dimension=embedding_dimension,
            distance_metric=distance_metric,
            extra_config=extra_config,
            **optional
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
