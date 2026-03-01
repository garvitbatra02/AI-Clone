"""
Embeddings Factory

Factory class for creating embedding model instances.
"""

from typing import Optional, Type

from RAGService.Data.Embeddings.base import (
    BaseEmbeddings,
    EmbeddingConfig,
    EmbeddingProvider,
)
from RAGService.Data.Embeddings.registry import (
    get_provider_class,
    is_provider_available,
    list_available_providers,
    register_provider,
)


class EmbeddingsFactory:
    """
    Factory for creating embedding model instances.
    
    Provides methods to create embedding instances from configuration
    objects or environment variables.
    
    Example:
        # Create from config
        config = EmbeddingConfig.for_cohere()
        embeddings = EmbeddingsFactory.create(config)
        
        # Create from environment
        embeddings = EmbeddingsFactory.create_from_env(
            provider=EmbeddingProvider.COHERE
        )
        
        # Create Cohere with defaults
        embeddings = EmbeddingsFactory.create_cohere()
    """
    
    @staticmethod
    def create(config: EmbeddingConfig) -> BaseEmbeddings:
        """
        Create an embedding model instance from configuration.
        
        Args:
            config: EmbeddingConfig with model details
            
        Returns:
            Configured BaseEmbeddings instance
            
        Raises:
            ValueError: If the provider is not available
        """
        provider_class = get_provider_class(config.provider)
        return provider_class(config)
    
    @staticmethod
    def create_from_env(
        provider: EmbeddingProvider,
        model_name: Optional[str] = None,
        **extra_config
    ) -> BaseEmbeddings:
        """
        Create an embedding model instance from environment variables.
        
        The provider implementation resolves its own API key from
        environment variables when not explicitly provided in config.
        
        Args:
            provider: The embedding provider
            model_name: Model name (uses provider default if not provided)
            **extra_config: Additional configuration
            
        Returns:
            Configured BaseEmbeddings instance
        """
        # Default model names per provider
        default_models = {
            EmbeddingProvider.COHERE: "embed-english-v3.0",
            EmbeddingProvider.OPENAI: "text-embedding-3-small",
            EmbeddingProvider.HUGGINGFACE: "sentence-transformers/all-mpnet-base-v2",
        }
        
        model = model_name or default_models.get(provider, "")
        
        config = EmbeddingConfig(
            provider=provider,
            model_name=model,
            extra_config=extra_config
        )
        
        return EmbeddingsFactory.create(config)
    
    @staticmethod
    def create_cohere(
        model_name: str = "embed-english-v3.0",
        **extra_config
    ) -> BaseEmbeddings:
        """
        Convenience method to create Cohere embeddings.
        
        The Cohere provider resolves its own API key from the
        COHERE_API_KEY environment variable when not provided in config.
        
        Args:
            model_name: Model name (default: embed-english-v3.0)
            **extra_config: Additional configuration
            
        Returns:
            Configured CohereEmbeddings instance
        """
        config = EmbeddingConfig.for_cohere(
            model_name=model_name,
            **extra_config
        )
        return EmbeddingsFactory.create(config)
    
    @staticmethod
    def create_openai(
        model_name: str = "text-embedding-3-small",
        **extra_config
    ) -> BaseEmbeddings:
        """
        Convenience method to create OpenAI embeddings.
        
        The OpenAI provider resolves its own API key from the
        OPENAI_API_KEY environment variable when not provided in config.
        
        Args:
            model_name: Model name (default: text-embedding-3-small)
            **extra_config: Additional configuration
            
        Returns:
            Configured OpenAIEmbeddings instance
        """
        config = EmbeddingConfig.for_openai(
            model_name=model_name,
            **extra_config
        )
        return EmbeddingsFactory.create(config)
    
    @staticmethod
    def create_huggingface(
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        **extra_config
    ) -> BaseEmbeddings:
        """
        Convenience method to create HuggingFace embeddings.
        
        Args:
            model_name: Model name (default: sentence-transformers/all-mpnet-base-v2)
            **extra_config: Additional configuration
            
        Returns:
            Configured HuggingFaceEmbeddings instance
        """
        config = EmbeddingConfig.for_huggingface(
            model_name=model_name,
            **extra_config
        )
        return EmbeddingsFactory.create(config)
    
    @staticmethod
    def register_provider(
        provider: EmbeddingProvider,
        implementation: Type[BaseEmbeddings],
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
    def list_providers() -> list[EmbeddingProvider]:
        """
        List all available providers.
        
        Returns:
            List of available EmbeddingProvider enums
        """
        return list_available_providers()
    
    @staticmethod
    def is_available(provider: EmbeddingProvider) -> bool:
        """
        Check if a provider is available.
        
        Args:
            provider: The provider to check
            
        Returns:
            True if the provider is registered and available
        """
        return is_provider_available(provider)
