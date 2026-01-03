"""
Chat Service - Handles LLM inference with provider rotation.

This module provides a centralized service for chat inference that:
- Rotates between multiple LLM providers (Groq, Cerebras) to handle rate limits
- Maintains singleton LLM instances per provider (initialized once)
- Provides a simple interface for FastAPI server to call

Usage:
    from ChatServer.services import chat_inference, get_chat_service
    
    # Simple usage - uses global service with auto-rotation
    response = chat_inference(session)
    
    # Or get the service instance for more control
    service = get_chat_service()
    response = service.chat(session)
"""

import os
import logging
from typing import Optional, Iterator, AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
import threading

from ..llm.base import BaseLLM, LLMConfig, LLMResponse, LLMProvider

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
from ..llm.factory import LLMFactory
from ..session.chat_session import ChatSession


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider."""
    provider: LLMProvider
    model: str
    api_key_env: str  # Environment variable name for API key
    
    def get_api_key(self) -> str:
        """Get API key from environment variable."""
        return os.getenv(self.api_key_env, "")


# Default provider configurations for rotation
DEFAULT_PROVIDERS = [
    ProviderConfig(
        provider=LLMProvider.GROQ,
        model="llama-3.3-70b-versatile",
        api_key_env="GROQ_API_KEY",
    ),
    ProviderConfig(
        provider=LLMProvider.CEREBRAS,
        model="llama-3.3-70b",
        api_key_env="CEREBRAS_API_KEY",
    ),
]


class ChatService:
    """
    Chat service that manages LLM instances and rotates between providers.
    
    This service:
    - Initializes LLM instances lazily (on first use)
    - Caches LLM instances for reuse (singleton per provider)
    - Rotates between providers on each request to distribute rate limits
    - Thread-safe for concurrent requests
    
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
    ):
        """
        Initialize the chat service.
        
        Args:
            providers: List of provider configurations. Defaults to Groq and Cerebras.
            temperature: Temperature for LLM responses.
            max_tokens: Maximum tokens for responses (None for model default).
        """
        self._providers = providers or DEFAULT_PROVIDERS
        self._temperature = temperature
        self._max_tokens = max_tokens
        
        # Cache for initialized LLM instances (provider -> LLM instance)
        self._llm_cache: dict[LLMProvider, BaseLLM] = {}
        
        # Current provider index for rotation
        self._current_index = 0
        
        # Thread lock for thread-safe rotation
        self._lock = threading.Lock()
    
    def _get_or_create_llm(self, config: ProviderConfig) -> BaseLLM:
        """
        Get cached LLM instance or create a new one.
        
        Args:
            config: Provider configuration.
            
        Returns:
            Initialized LLM instance.
        """
        if config.provider not in self._llm_cache:
            llm = LLMFactory.create(
                provider=config.provider,
                model=config.model,
                api_key=config.get_api_key(),
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
            self._llm_cache[config.provider] = llm
        
        return self._llm_cache[config.provider]
    
    def _get_next_provider(self) -> ProviderConfig:
        """
        Get the next provider in rotation (thread-safe).
        
        Returns:
            Next provider configuration.
        """
        with self._lock:
            config = self._providers[self._current_index]
            self._current_index = (self._current_index + 1) % len(self._providers)
            return config
    
    def get_current_provider(self) -> ProviderConfig:
        """
        Get the current provider without rotating.
        
        Returns:
            Current provider configuration.
        """
        with self._lock:
            return self._providers[self._current_index]
    
    def chat(
        self,
        session: ChatSession,
        provider: Optional[LLMProvider] = None,
        fallback: bool = False,
    ) -> LLMResponse:
        """
        Send a chat request and get a response.
        
        If provider is not specified, uses the next provider in rotation.
        
        Args:
            session: Chat session with conversation history.
            provider: Optional specific provider to use (skips rotation).
            fallback: If True, automatically try next providers on failure.
            
        Returns:
            LLM response with content and token usage.
            
        Raises:
            AllProvidersFailedError: If fallback=True and all providers fail.
        """
        if fallback and not provider:
            return self._chat_with_fallback(session)
        
        if provider:
            # Use specific provider
            config = next(
                (p for p in self._providers if p.provider == provider),
                self._providers[0]
            )
        else:
            # Use rotation
            config = self._get_next_provider()
        
        llm = self._get_or_create_llm(config)
        return llm.chat(session)
    
    def _chat_with_fallback(self, session: ChatSession) -> LLMResponse:
        """
        Try chat with automatic fallback to next providers on failure.
        
        Tries each provider once until one succeeds or all fail.
        
        Args:
            session: Chat session with conversation history.
            
        Returns:
            LLM response from the first successful provider.
            
        Raises:
            AllProvidersFailedError: If all providers fail.
        """
        errors: dict[str, str] = {}
        tried_providers: set[LLMProvider] = set()
        
        while len(tried_providers) < len(self._providers):
            config = self._get_next_provider()
            
            # Skip if already tried
            if config.provider in tried_providers:
                continue
            
            tried_providers.add(config.provider)
            
            try:
                llm = self._get_or_create_llm(config)
                response = llm.chat(session)
                logger.info(f"Successfully served request using {config.provider.value}")
                return response
            except Exception as e:
                error_msg = str(e)
                errors[config.provider.value] = error_msg
                logger.warning(
                    f"Provider {config.provider.value} failed: {error_msg}. "
                    f"Trying next provider..."
                )
        
        # All providers failed
        logger.error(f"All {len(self._providers)} providers failed")
        raise AllProvidersFailedError(errors)
    
    def chat_stream(
        self,
        session: ChatSession,
        provider: Optional[LLMProvider] = None,
        fallback: bool = False,
    ) -> Iterator[str]:
        """
        Send a chat request and stream the response.
        
        Args:
            session: Chat session with conversation history.
            provider: Optional specific provider to use (skips rotation).
            fallback: If True, automatically try next providers on failure.
            
        Yields:
            Response chunks as they arrive.
            
        Raises:
            AllProvidersFailedError: If fallback=True and all providers fail.
        """
        if fallback and not provider:
            yield from self._chat_stream_with_fallback(session)
            return
        
        if provider:
            config = next(
                (p for p in self._providers if p.provider == provider),
                self._providers[0]
            )
        else:
            config = self._get_next_provider()
        
        llm = self._get_or_create_llm(config)
        yield from llm.chat_stream(session)
    
    def _chat_stream_with_fallback(self, session: ChatSession) -> Iterator[str]:
        """
        Try streaming chat with automatic fallback on failure.
        
        Args:
            session: Chat session with conversation history.
            
        Yields:
            Response chunks from the first successful provider.
            
        Raises:
            AllProvidersFailedError: If all providers fail.
        """
        errors: dict[str, str] = {}
        tried_providers: set[LLMProvider] = set()
        
        while len(tried_providers) < len(self._providers):
            config = self._get_next_provider()
            
            if config.provider in tried_providers:
                continue
            
            tried_providers.add(config.provider)
            
            try:
                llm = self._get_or_create_llm(config)
                # Try to get first chunk to verify connection works
                stream = llm.chat_stream(session)
                first_chunk = next(stream)
                logger.info(f"Successfully streaming using {config.provider.value}")
                yield first_chunk
                yield from stream
                return
            except Exception as e:
                error_msg = str(e)
                errors[config.provider.value] = error_msg
                logger.warning(
                    f"Provider {config.provider.value} failed: {error_msg}. "
                    f"Trying next provider..."
                )
        
        logger.error(f"All {len(self._providers)} providers failed for streaming")
        raise AllProvidersFailedError(errors)
    
    async def chat_async(
        self,
        session: ChatSession,
        provider: Optional[LLMProvider] = None,
        fallback: bool = False,
    ) -> LLMResponse:
        """
        Send an async chat request and get a response.
        
        Args:
            session: Chat session with conversation history.
            provider: Optional specific provider to use (skips rotation).
            fallback: If True, automatically try next providers on failure.
            
        Returns:
            LLM response with content and token usage.
            
        Raises:
            AllProvidersFailedError: If fallback=True and all providers fail.
        """
        if fallback and not provider:
            return await self._chat_async_with_fallback(session)
        
        if provider:
            config = next(
                (p for p in self._providers if p.provider == provider),
                self._providers[0]
            )
        else:
            config = self._get_next_provider()
        
        llm = self._get_or_create_llm(config)
        return await llm.chat_async(session)
    
    async def _chat_async_with_fallback(self, session: ChatSession) -> LLMResponse:
        """
        Try async chat with automatic fallback on failure.
        
        Args:
            session: Chat session with conversation history.
            
        Returns:
            LLM response from the first successful provider.
            
        Raises:
            AllProvidersFailedError: If all providers fail.
        """
        errors: dict[str, str] = {}
        tried_providers: set[LLMProvider] = set()
        
        while len(tried_providers) < len(self._providers):
            config = self._get_next_provider()
            
            if config.provider in tried_providers:
                continue
            
            tried_providers.add(config.provider)
            
            try:
                llm = self._get_or_create_llm(config)
                response = await llm.chat_async(session)
                logger.info(f"Successfully served async request using {config.provider.value}")
                return response
            except Exception as e:
                error_msg = str(e)
                errors[config.provider.value] = error_msg
                logger.warning(
                    f"Provider {config.provider.value} failed: {error_msg}. "
                    f"Trying next provider..."
                )
        
        logger.error(f"All {len(self._providers)} providers failed for async")
        raise AllProvidersFailedError(errors)
    
    async def chat_stream_async(
        self,
        session: ChatSession,
        provider: Optional[LLMProvider] = None,
        fallback: bool = False,
    ) -> AsyncIterator[str]:
        """
        Send an async chat request and stream the response.
        
        Args:
            session: Chat session with conversation history.
            provider: Optional specific provider to use (skips rotation).
            fallback: If True, automatically try next providers on failure.
            
        Yields:
            Response chunks as they arrive.
            
        Raises:
            AllProvidersFailedError: If fallback=True and all providers fail.
        """
        if fallback and not provider:
            async for chunk in self._chat_stream_async_with_fallback(session):
                yield chunk
            return
        
        if provider:
            config = next(
                (p for p in self._providers if p.provider == provider),
                self._providers[0]
            )
        else:
            config = self._get_next_provider()
        
        llm = self._get_or_create_llm(config)
        async for chunk in llm.chat_stream_async(session):
            yield chunk
    
    async def _chat_stream_async_with_fallback(
        self, session: ChatSession
    ) -> AsyncIterator[str]:
        """
        Try async streaming chat with automatic fallback on failure.
        
        Args:
            session: Chat session with conversation history.
            
        Yields:
            Response chunks from the first successful provider.
            
        Raises:
            AllProvidersFailedError: If all providers fail.
        """
        errors: dict[str, str] = {}
        tried_providers: set[LLMProvider] = set()
        
        while len(tried_providers) < len(self._providers):
            config = self._get_next_provider()
            
            if config.provider in tried_providers:
                continue
            
            tried_providers.add(config.provider)
            
            try:
                llm = self._get_or_create_llm(config)
                stream = llm.chat_stream_async(session)
                # Try to get first chunk to verify connection works
                first_chunk = await stream.__anext__()
                logger.info(f"Successfully streaming async using {config.provider.value}")
                yield first_chunk
                async for chunk in stream:
                    yield chunk
                return
            except Exception as e:
                error_msg = str(e)
                errors[config.provider.value] = error_msg
                logger.warning(
                    f"Provider {config.provider.value} failed: {error_msg}. "
                    f"Trying next provider..."
                )
        
        logger.error(f"All {len(self._providers)} providers failed for async streaming")
        raise AllProvidersFailedError(errors)
    
    def add_provider(self, config: ProviderConfig) -> None:
        """
        Add a new provider to the rotation.
        
        Args:
            config: Provider configuration to add.
        """
        with self._lock:
            # Avoid duplicates
            if not any(p.provider == config.provider for p in self._providers):
                self._providers.append(config)
    
    def remove_provider(self, provider: LLMProvider) -> None:
        """
        Remove a provider from the rotation.
        
        Args:
            provider: Provider to remove.
        """
        with self._lock:
            self._providers = [p for p in self._providers if p.provider != provider]
            # Remove from cache if exists
            if provider in self._llm_cache:
                del self._llm_cache[provider]
            # Reset index if out of bounds
            if self._current_index >= len(self._providers):
                self._current_index = 0
    
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


# Global singleton instance
_chat_service: Optional[ChatService] = None
_service_lock = threading.Lock()


def get_chat_service(
    providers: Optional[list[ProviderConfig]] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    force_new: bool = False,
) -> ChatService:
    """
    Get the global chat service instance (singleton).
    
    Args:
        providers: Provider configurations (only used on first call or force_new).
        temperature: Temperature setting (only used on first call or force_new).
        max_tokens: Max tokens setting (only used on first call or force_new).
        force_new: If True, create a new instance even if one exists.
        
    Returns:
        Global ChatService instance.
    """
    global _chat_service
    
    with _service_lock:
        if _chat_service is None or force_new:
            _chat_service = ChatService(
                providers=providers,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        return _chat_service


def chat_inference(
    session: ChatSession,
    provider: Optional[LLMProvider] = None,
    fallback: bool = True,
) -> LLMResponse:
    """
    Simple function to perform chat inference using the global service.
    
    This is the primary function to be called by the FastAPI server.
    Uses automatic provider rotation for rate limit management.
    
    Args:
        session: Chat session with conversation history.
        provider: Optional specific provider (skips rotation if provided).
        fallback: If True (default), automatically tries next providers on failure.
                  Tries all providers once before raising an error.
        
    Returns:
        LLM response with content and token usage.
        
    Raises:
        AllProvidersFailedError: If fallback=True and all providers fail.
        
    Example:
        from ChatServer.services import chat_inference
        from ChatServer import ChatSession
        
        session = ChatSession()
        session.add_user_message("Hello!")
        
        # With fallback (default) - tries all providers before failing
        response = chat_inference(session)
        
        # Without fallback - fails immediately on error
        response = chat_inference(session, fallback=False)
        
        print(response.content)
    """
    service = get_chat_service()
    return service.chat(session, provider=provider, fallback=fallback)


def chat_inference_stream(
    session: ChatSession,
    provider: Optional[LLMProvider] = None,
    fallback: bool = True,
) -> Iterator[str]:
    """
    Stream chat inference using the global service.
    
    Args:
        session: Chat session with conversation history.
        provider: Optional specific provider (skips rotation if provided).
        fallback: If True (default), automatically tries next providers on failure.
        
    Yields:
        Response chunks as they arrive.
        
    Raises:
        AllProvidersFailedError: If fallback=True and all providers fail.
    """
    service = get_chat_service()
    yield from service.chat_stream(session, provider=provider, fallback=fallback)


async def chat_inference_async(
    session: ChatSession,
    provider: Optional[LLMProvider] = None,
    fallback: bool = True,
) -> LLMResponse:
    """
    Async chat inference using the global service.
    
    Args:
        session: Chat session with conversation history.
        provider: Optional specific provider (skips rotation if provided).
        fallback: If True (default), automatically tries next providers on failure.
        
    Returns:
        LLM response with content and token usage.
        
    Raises:
        AllProvidersFailedError: If fallback=True and all providers fail.
    """
    service = get_chat_service()
    return await service.chat_async(session, provider=provider, fallback=fallback)


async def chat_inference_stream_async(
    session: ChatSession,
    provider: Optional[LLMProvider] = None,
    fallback: bool = True,
) -> AsyncIterator[str]:
    """
    Async streaming chat inference using the global service.
    
    Args:
        session: Chat session with conversation history.
        provider: Optional specific provider (skips rotation if provided).
        fallback: If True (default), automatically tries next providers on failure.
        
    Yields:
        Response chunks as they arrive.
        
    Raises:
        AllProvidersFailedError: If fallback=True and all providers fail.
    """
    service = get_chat_service()
    async for chunk in service.chat_stream_async(session, provider=provider, fallback=fallback):
        yield chunk
