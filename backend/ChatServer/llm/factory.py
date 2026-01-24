"""
LLM Factory for creating LLM instances.

Provides a unified interface for creating LLM instances based on provider type.
Supports both single API key and multiple API keys for resilience.
"""

from typing import Optional, Type, Union
from .base import BaseLLM, LLMConfig, LLMProvider
from .proprietary_llms.gemini_llm import GeminiLLM
from .proprietary_llms.groq_llm import GroqLLM
from .proprietary_llms.cerebras_llm import CerebrasLLM
from .model_registry import get_provider_for_model


class LLMFactory:
    """
    Factory class for creating LLM instances with built-in resilience.
    
    This class provides a centralized way to create LLM instances
    based on the provider type or model name. Supports multiple API keys
    for automatic key rotation and retry on failure.
    
    Example:
        # Create with single API key
        llm = LLMFactory.create(
            provider=LLMProvider.GEMINI,
            model="gemini-2.5-flash",
            api_key="your-api-key"
        )
        
        # Create with multiple API keys (resilient)
        llm = LLMFactory.create(
            provider=LLMProvider.GROQ,
            model="llama-3.3-70b-versatile",
            api_keys=["key1", "key2", "key3"]  # Auto-rotation on failure
        )
    """
    
    # Mapping of providers to their LLM classes
    _provider_map: dict[LLMProvider, Type[BaseLLM]] = {
        LLMProvider.GEMINI: GeminiLLM,
        LLMProvider.GROQ: GroqLLM,
        LLMProvider.CEREBRAS: CerebrasLLM,
    }
    
    @classmethod
    def create(
        cls,
        provider: LLMProvider,
        model: str,
        api_key: Optional[str] = None,
        api_keys: Optional[list[str]] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        max_retries: int = 2,
        **kwargs,
    ) -> BaseLLM:
        """
        Create an LLM instance for the specified provider.
        
        Args:
            provider: The LLM provider
            model: The model identifier
            api_key: Single API key (for backward compatibility)
            api_keys: List of API keys for resilience (preferred)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            max_retries: Max retry attempts on empty response
            **kwargs: Additional configuration parameters
            
        Returns:
            An initialized LLM instance with built-in resilience
            
        Raises:
            ValueError: If the provider is not supported or no API keys provided
        """
        if provider not in cls._provider_map:
            raise ValueError(f"Unsupported provider: {provider}")
        
        # Handle API keys - support both single key and list
        if api_keys is None and api_key is None:
            raise ValueError("Either api_key or api_keys must be provided")
        
        if api_keys is None:
            api_keys = [api_key]
        
        config = LLMConfig(
            model=model,
            api_keys=api_keys,
            max_retries=max_retries,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        
        llm_class = cls._provider_map[provider]
        return llm_class(config)
    
    @classmethod
    def from_model(
        cls,
        model: str,
        api_key: Optional[str] = None,
        api_keys: Optional[list[str]] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        max_retries: int = 2,
        provider: Optional[LLMProvider] = None,
        **kwargs,
    ) -> BaseLLM:
        """
        Create an LLM instance by auto-detecting the provider from the model name.
        
        Args:
            model: The model identifier
            api_key: Single API key (for backward compatibility)
            api_keys: List of API keys for resilience (preferred)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            max_retries: Max retry attempts on empty response
            provider: Optional provider override (auto-detected if not provided)
            **kwargs: Additional configuration parameters
            
        Returns:
            An initialized LLM instance with built-in resilience
            
        Raises:
            ValueError: If the provider cannot be determined
        """
        if provider is None:
            provider = cls._detect_provider(model)
        
        if provider is None:
            raise ValueError(
                f"Cannot auto-detect provider for model: {model}. "
                "Please specify the provider explicitly."
            )
        
        return cls.create(
            provider=provider,
            model=model,
            api_key=api_key,
            api_keys=api_keys,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            **kwargs,
        )
    
    @classmethod
    def _detect_provider(cls, model: str) -> Optional[LLMProvider]:
        """
        Auto-detect the provider based on model name using the model registry.
        
        This performs a deterministic lookup in the model registry instead of
        prefix matching, ensuring accurate provider detection.
        
        Args:
            model: The exact model identifier
            
        Returns:
            The detected provider or None if model is not in registry
        """
        return get_provider_for_model(model)
    
    @classmethod
    def register_provider(
        cls,
        provider: LLMProvider,
        llm_class: Type[BaseLLM],
    ) -> None:
        """
        Register a custom LLM provider.
        
        Args:
            provider: The provider identifier
            llm_class: The LLM class to register
        """
        cls._provider_map[provider] = llm_class
    
    @classmethod
    def register_model(cls, model: str, provider: LLMProvider) -> None:
        """
        Register a model for auto-detection.
        
        This dynamically adds a model to the model registry at runtime.
        
        Args:
            model: The exact model name
            provider: The associated provider
        """
        from .model_registry import MODEL_REGISTRY
        MODEL_REGISTRY[model] = provider
    
    @classmethod
    def get_supported_providers(cls) -> list[LLMProvider]:
        """
        Get a list of supported providers.
        
        Returns:
            List of supported LLMProvider values
        """
        return list(cls._provider_map.keys())

