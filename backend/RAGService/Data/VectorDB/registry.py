"""
VectorDB Registry

Maps provider names to their implementations for auto-discovery.
"""

from typing import Dict, Type

from RAGService.Data.VectorDB.base import BaseVectorDB, VectorDBProvider


# Provider implementation registry
# Populated by register_provider() or lazily on first access
_PROVIDER_REGISTRY: Dict[VectorDBProvider, Type[BaseVectorDB]] = {}


def _load_default_providers() -> None:
    """Load default provider implementations."""
    global _PROVIDER_REGISTRY
    
    if _PROVIDER_REGISTRY:
        return  # Already loaded
    
    # Import providers lazily to avoid circular imports
    try:
        from RAGService.Data.VectorDB.providers.qdrant_db import QdrantVectorDB
        _PROVIDER_REGISTRY[VectorDBProvider.QDRANT] = QdrantVectorDB
    except ImportError:
        pass  # Qdrant not installed
    
    # Add more providers here as they are implemented
    # try:
    #     from RAGService.Data.VectorDB.providers.pinecone_db import PineconeVectorDB
    #     _PROVIDER_REGISTRY[VectorDBProvider.PINECONE] = PineconeVectorDB
    # except ImportError:
    #     pass


def get_provider_class(provider: VectorDBProvider) -> Type[BaseVectorDB]:
    """
    Get the implementation class for a provider.
    
    Args:
        provider: The vector database provider
        
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
    provider: VectorDBProvider,
    implementation: Type[BaseVectorDB],
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


def list_available_providers() -> list[VectorDBProvider]:
    """
    List all available (registered) providers.
    
    Returns:
        List of registered VectorDBProvider enums
    """
    _load_default_providers()
    return list(_PROVIDER_REGISTRY.keys())


def is_provider_available(provider: VectorDBProvider) -> bool:
    """
    Check if a provider is available.
    
    Args:
        provider: The provider to check
        
    Returns:
        True if the provider is registered
    """
    _load_default_providers()
    return provider in _PROVIDER_REGISTRY
