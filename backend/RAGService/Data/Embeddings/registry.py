"""
Embeddings Registry

Maps provider names to their implementations for auto-discovery.
"""

from typing import Dict, Type

from RAGService.Data.Embeddings.base import BaseEmbeddings, EmbeddingProvider


# Provider implementation registry
_PROVIDER_REGISTRY: Dict[EmbeddingProvider, Type[BaseEmbeddings]] = {}


def _load_default_providers() -> None:
    """Load default provider implementations."""
    global _PROVIDER_REGISTRY
    
    if _PROVIDER_REGISTRY:
        return  # Already loaded
    
    # Import providers lazily to avoid circular imports
    try:
        from RAGService.Data.Embeddings.providers.cohere_embeddings import CohereEmbeddings
        _PROVIDER_REGISTRY[EmbeddingProvider.COHERE] = CohereEmbeddings
    except ImportError:
        pass  # Cohere not installed
    
    try:
        from RAGService.Data.Embeddings.providers.openai_embeddings import OpenAIEmbeddings
        _PROVIDER_REGISTRY[EmbeddingProvider.OPENAI] = OpenAIEmbeddings
    except ImportError:
        pass  # OpenAI not installed
    
    try:
        from RAGService.Data.Embeddings.providers.huggingface_embeddings import HuggingFaceEmbeddings
        _PROVIDER_REGISTRY[EmbeddingProvider.HUGGINGFACE] = HuggingFaceEmbeddings
    except ImportError:
        pass  # HuggingFace not installed


def get_provider_class(provider: EmbeddingProvider) -> Type[BaseEmbeddings]:
    """
    Get the implementation class for a provider.
    
    Args:
        provider: The embedding provider
        
    Returns:
        The implementation class for the provider
        
    Raises:
        ValueError: If the provider is not registered
    """
    _load_default_providers()
    
    if provider not in _PROVIDER_REGISTRY:
        available = list(_PROVIDER_REGISTRY.keys())
        raise ValueError(
            f"Provider '{provider.value}' is not registered. "
            f"Available providers: {[p.value for p in available]}"
        )
    
    return _PROVIDER_REGISTRY[provider]


def register_provider(
    provider: EmbeddingProvider,
    implementation: Type[BaseEmbeddings],
    override: bool = False
) -> None:
    """
    Register a provider implementation.
    
    Args:
        provider: The provider to register
        implementation: The implementation class
        override: Whether to override existing registration
        
    Raises:
        ValueError: If provider already registered and override is False
    """
    if provider in _PROVIDER_REGISTRY and not override:
        raise ValueError(
            f"Provider '{provider.value}' is already registered. "
            "Use override=True to replace."
        )
    
    _PROVIDER_REGISTRY[provider] = implementation


def list_available_providers() -> list[EmbeddingProvider]:
    """
    List all available (registered) providers.
    
    Returns:
        List of registered EmbeddingProvider enums
    """
    _load_default_providers()
    return list(_PROVIDER_REGISTRY.keys())


def is_provider_available(provider: EmbeddingProvider) -> bool:
    """
    Check if a provider is available.
    
    Args:
        provider: The provider to check
        
    Returns:
        True if the provider is registered
    """
    _load_default_providers()
    return provider in _PROVIDER_REGISTRY
