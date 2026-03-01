"""
Chat Service - Simplified provider rotation with resilience delegated to LLM layer.

This module provides a centralized service for chat inference that:
- Rotates between multiple LLM providers (Groq, Cerebras) to handle rate limits
- Delegates key rotation and retry logic to the LLM layer (BaseLLM)
- Provides a simple interface for FastAPI server to call

API keys are fully encapsulated inside the LLM layer. This service
never touches, passes, or even knows about API keys.

Usage:
    from ChatServer.services import chat_inference, get_chat_service
    
    # Simple usage - uses global service with auto-rotation
    response = chat_inference(session)
    
    # Or get the service instance for more control
    service = get_chat_service()
    response = service.chat(session)
"""

import logging
from typing import Optional, Iterator, AsyncIterator
from dataclasses import dataclass
import threading

from ..llm.base import BaseLLM, LLMConfig, LLMResponse, LLMProvider, AllKeysFailedError
from ..llm.model_registry import get_provider_for_model
from ..llm.factory import LLMFactory
from ..session.chat_session import ChatSession

# Configure logging
logger = logging.getLogger(__name__)


class AllProvidersFailedError(Exception):
    """
    Raised when all LLM providers fail to serve a request.
    
    Attributes:
        errors: Dictionary mapping provider names to their error messages.
    """
    def __init__(self, errors: dict[str, str]):
        self.errors = errors
        provider_errors = "; ".join([f"{k}: {v}" for k, v in errors.items()])
        super().__init__(f"All providers failed. Errors: {provider_errors}")


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider."""
    provider: LLMProvider
    default_model: str


# Default provider configurations for rotation
DEFAULT_PROVIDERS = [
    ProviderConfig(
        provider=LLMProvider.GROQ,
        default_model="llama-3.3-70b-versatile",
    ),
    ProviderConfig(
        provider=LLMProvider.CEREBRAS,
        default_model="llama3.1-8b",
    ),
]


class ChatService:
    """
    Chat service that manages LLM instances and rotates between providers.
    
    Key rotation and retry logic is handled by the LLM layer (BaseLLM).
    This service only handles provider-level rotation.
    
    Example:
        service = ChatService()
        
        # Each call rotates to next provider
        response1 = service.chat(session)  # Uses Groq
        response2 = service.chat(session)  # Uses Cerebras
        response3 = service.chat(session)  # Uses Groq again
    """
    
    def __init__(
        self,
        providers: Optional[list[ProviderConfig]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        max_retries: int = 2,
    ):
        """
        Initialize the chat service.
        
        Args:
            providers: List of provider configurations.
            temperature: Temperature for LLM responses.
            max_tokens: Maximum tokens for responses.
            max_retries: Max retry attempts on empty response (passed to LLM).
        """
        self._providers = providers or DEFAULT_PROVIDERS
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        
        # Cache for initialized LLM instances (provider -> LLM)
        self._llm_cache: dict[LLMProvider, BaseLLM] = {}
        
        # Current provider index for rotation
        self._current_index = 0
        
        # Thread lock for thread-safe rotation
        self._lock = threading.Lock()
    
    def _get_or_create_llm(self, config: ProviderConfig, model: Optional[str] = None) -> BaseLLM:
        """
        Get cached LLM instance or create a new one.
        
        LLM instances handle their own API key loading and rotation internally.
        
        Args:
            config: Provider configuration.
            model: Optional specific model (uses default if None).
            
        Returns:
            Initialized LLM instance with resilience.
        """
        model_to_use = model or config.default_model
        cache_key = (config.provider, model_to_use)
        
        if cache_key not in self._llm_cache:
            llm = LLMFactory.create(
                provider=config.provider,
                model=model_to_use,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                max_retries=self._max_retries,
            )
            self._llm_cache[cache_key] = llm
        
        return self._llm_cache[cache_key]
    
    def _get_next_provider(self) -> ProviderConfig:
        """Get the next provider in rotation (thread-safe)."""
        with self._lock:
            config = self._providers[self._current_index]
            self._current_index = (self._current_index + 1) % len(self._providers)
            return config
    
    def get_current_provider(self) -> ProviderConfig:
        """Get the current provider without rotating."""
        with self._lock:
            return self._providers[self._current_index]
    
    # ==================== Main Chat Methods ====================
    
    def chat(
        self,
        session: ChatSession,
        provider: Optional[LLMProvider] = None,
        fallback: bool = False,
        model: Optional[str] = None,
    ) -> LLMResponse:
        """
        Send a chat request and get a response.
        
        Key rotation and retry on empty response is handled by the LLM layer.
        This method only handles provider-level rotation.
        
        Args:
            session: Chat session with conversation history.
            provider: Optional specific provider to use.
            fallback: If True, try next providers on failure.
            model: Optional specific model name.
            
        Returns:
            LLM response with content and token usage.
        """
        # Auto-detect provider from model
        if model and not provider:
            provider = get_provider_for_model(model)
            if provider is None:
                raise ValueError(f"Model '{model}' not found in MODEL_REGISTRY")
        
        # Get provider config with rotation
        if provider:
            config = next(
                (p for p in self._providers if p.provider == provider),
                self._providers[0]
            )
        else:
            config = self._get_next_provider()
        
        # Try with fallback to other providers (starting from rotated provider)
        if fallback:
            return self._chat_with_provider_fallback(session, config, model)
        
        # Single provider attempt (LLM handles key rotation internally)
        llm = self._get_or_create_llm(config, model)
        return llm.chat(session)
    
    def _chat_with_provider_fallback(
        self, 
        session: ChatSession,
        starting_config: ProviderConfig,
        model: Optional[str] = None
    ) -> LLMResponse:
        """
        Try chat with automatic fallback to next providers.
        
        Starts from the rotated provider, then tries others on failure.
        Each LLM handles its own key rotation and retry internally.
        """
        errors: dict[str, str] = {}
        
        # Find starting index for rotation
        start_index = next(
            (i for i, p in enumerate(self._providers) if p.provider == starting_config.provider),
            0
        )
        
        # Try all providers starting from the rotated one
        for i in range(len(self._providers)):
            config = self._providers[(start_index + i) % len(self._providers)]
            try:
                llm = self._get_or_create_llm(config, model)
                response = llm.chat(session)
                logger.info(f"Successfully served request using {config.provider.value}")
                return response
            except AllKeysFailedError as e:
                errors[config.provider.value] = str(e)
                logger.warning(f"Provider {config.provider.value} failed: {e}")
            except Exception as e:
                errors[config.provider.value] = str(e)
                logger.warning(f"Provider {config.provider.value} failed: {e}")
        
        raise AllProvidersFailedError(errors)
    
    async def chat_async(
        self,
        session: ChatSession,
        provider: Optional[LLMProvider] = None,
        fallback: bool = False,
        model: Optional[str] = None,
    ) -> LLMResponse:
        """Async version of chat with provider rotation."""
        # Auto-detect provider from model
        if model and not provider:
            provider = get_provider_for_model(model)
            if provider is None:
                raise ValueError(f"Model '{model}' not found in MODEL_REGISTRY")
        
        if provider:
            config = next(
                (p for p in self._providers if p.provider == provider),
                self._providers[0]
            )
        else:
            config = self._get_next_provider()
        
        if fallback:
            return await self._chat_async_with_provider_fallback(session, config, model)
        
        llm = self._get_or_create_llm(config, model)
        return await llm.chat_async(session)
    
    async def _chat_async_with_provider_fallback(
        self, 
        session: ChatSession,
        starting_config: ProviderConfig,
        model: Optional[str] = None
    ) -> LLMResponse:
        """Async chat with provider-level fallback starting from rotated provider."""
        errors: dict[str, str] = {}
        
        # Find starting index for rotation
        start_index = next(
            (i for i, p in enumerate(self._providers) if p.provider == starting_config.provider),
            0
        )
        
        # Try all providers starting from the rotated one
        for i in range(len(self._providers)):
            config = self._providers[(start_index + i) % len(self._providers)]
            try:
                llm = self._get_or_create_llm(config, model)
                response = await llm.chat_async(session)
                logger.info(f"Successfully served async request using {config.provider.value}")
                return response
            except AllKeysFailedError as e:
                errors[config.provider.value] = str(e)
                logger.warning(f"Provider {config.provider.value} failed: {e}")
            except Exception as e:
                errors[config.provider.value] = str(e)
                logger.warning(f"Provider {config.provider.value} failed: {e}")
        
        raise AllProvidersFailedError(errors)
    
    def chat_stream(
        self,
        session: ChatSession,
        provider: Optional[LLMProvider] = None,
        fallback: bool = False,
        model: Optional[str] = None,
    ) -> Iterator[str]:
        """Stream chat with provider rotation."""
        if model and not provider:
            provider = get_provider_for_model(model)
            if provider is None:
                raise ValueError(f"Model '{model}' not found in MODEL_REGISTRY")
        
        if provider:
            config = next(
                (p for p in self._providers if p.provider == provider),
                self._providers[0]
            )
        else:
            config = self._get_next_provider()
        
        if fallback:
            yield from self._chat_stream_with_provider_fallback(session, config, model)
            return
        
        llm = self._get_or_create_llm(config, model)
        yield from llm.chat_stream(session)
    
    def _chat_stream_with_provider_fallback(
        self, 
        session: ChatSession,
        starting_config: ProviderConfig,
        model: Optional[str] = None
    ) -> Iterator[str]:
        """Stream chat with provider-level fallback starting from rotated provider."""
        errors: dict[str, str] = {}
        
        # Find starting index for rotation
        start_index = next(
            (i for i, p in enumerate(self._providers) if p.provider == starting_config.provider),
            0
        )
        
        # Try all providers starting from the rotated one
        for i in range(len(self._providers)):
            config = self._providers[(start_index + i) % len(self._providers)]
            try:
                llm = self._get_or_create_llm(config, model)
                yield from llm.chat_stream(session)
                return
            except AllKeysFailedError as e:
                errors[config.provider.value] = str(e)
                logger.warning(f"Provider {config.provider.value} failed: {e}")
            except Exception as e:
                errors[config.provider.value] = str(e)
                logger.warning(f"Provider {config.provider.value} failed: {e}")
        
        raise AllProvidersFailedError(errors)
    
    async def chat_stream_async(
        self,
        session: ChatSession,
        provider: Optional[LLMProvider] = None,
        fallback: bool = False,
        model: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Async stream chat with provider rotation."""
        if model and not provider:
            provider = get_provider_for_model(model)
            if provider is None:
                raise ValueError(f"Model '{model}' not found in MODEL_REGISTRY")
        
        if provider:
            config = next(
                (p for p in self._providers if p.provider == provider),
                self._providers[0]
            )
        else:
            config = self._get_next_provider()
        
        if fallback:
            async for chunk in self._chat_stream_async_with_provider_fallback(session, config, model):
                yield chunk
            return
        
        llm = self._get_or_create_llm(config, model)
        async for chunk in llm.chat_stream_async(session):
            yield chunk
    
    async def _chat_stream_async_with_provider_fallback(
        self, 
        session: ChatSession,
        starting_config: ProviderConfig,
        model: Optional[str] = None
    ) -> AsyncIterator[str]:
        """Async stream chat with provider-level fallback starting from rotated provider."""
        errors: dict[str, str] = {}
        
        # Find starting index for rotation
        start_index = next(
            (i for i, p in enumerate(self._providers) if p.provider == starting_config.provider),
            0
        )
        
        # Try all providers starting from the rotated one
        for i in range(len(self._providers)):
            config = self._providers[(start_index + i) % len(self._providers)]
            try:
                llm = self._get_or_create_llm(config, model)
                async for chunk in llm.chat_stream_async(session):
                    yield chunk
                return
            except AllKeysFailedError as e:
                errors[config.provider.value] = str(e)
                logger.warning(f"Provider {config.provider.value} failed: {e}")
            except Exception as e:
                errors[config.provider.value] = str(e)
                logger.warning(f"Provider {config.provider.value} failed: {e}")
        
        raise AllProvidersFailedError(errors)
    
    # ==================== Utility Methods ====================
    
    def get_provider_count(self) -> int:
        """Get the number of providers in rotation."""
        return len(self._providers)
    
    def get_active_providers(self) -> list[LLMProvider]:
        """Get list of active providers."""
        return [p.provider for p in self._providers]
    
    def clear_cache(self) -> None:
        """Clear all cached LLM instances."""
        with self._lock:
            self._llm_cache.clear()


# ==================== Global Singleton ====================

_chat_service: Optional[ChatService] = None
_service_lock = threading.Lock()


def get_chat_service(
    providers: Optional[list[ProviderConfig]] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    force_new: bool = False,
) -> ChatService:
    """Get the global chat service instance (singleton)."""
    global _chat_service
    
    with _service_lock:
        if _chat_service is None or force_new:
            _chat_service = ChatService(
                providers=providers,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        return _chat_service


# ==================== Convenience Functions ====================

def chat_inference(
    session: ChatSession,
    provider: Optional[LLMProvider] = None,
    fallback: bool = True,
    model: Optional[str] = None,
) -> LLMResponse:
    """Simple function to perform chat inference using the global service."""
    service = get_chat_service()
    return service.chat(session, provider=provider, fallback=fallback, model=model)


def chat_inference_stream(
    session: ChatSession,
    provider: Optional[LLMProvider] = None,
    fallback: bool = True,
    model: Optional[str] = None,
) -> Iterator[str]:
    """Stream chat inference using the global service."""
    service = get_chat_service()
    yield from service.chat_stream(session, provider=provider, fallback=fallback, model=model)


async def chat_inference_async(
    session: ChatSession,
    provider: Optional[LLMProvider] = None,
    fallback: bool = True,
    model: Optional[str] = None,
) -> LLMResponse:
    """Async chat inference using the global service."""
    service = get_chat_service()
    return await service.chat_async(session, provider=provider, fallback=fallback, model=model)


async def chat_inference_stream_async(
    session: ChatSession,
    provider: Optional[LLMProvider] = None,
    fallback: bool = True,
    model: Optional[str] = None,
) -> AsyncIterator[str]:
    """Async streaming chat inference using the global service."""
    service = get_chat_service()
    async for chunk in service.chat_stream_async(session, provider=provider, fallback=fallback, model=model):
        yield chunk
