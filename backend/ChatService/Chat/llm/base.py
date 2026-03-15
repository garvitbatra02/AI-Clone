"""
Base LLM class providing the abstract interface for all LLM providers.

This module defines the contract that all LLM implementations must follow,
ensuring a consistent API across different providers.

Resilience features are inherited from ``shared.key_rotation.KeyRotationMixin``:
- Automatic API key loading from environment variables
- Multiple API keys with automatic rotation
- Retry on failure or empty response
- Per-key client caching (sync + async)
- All LLM implementations inherit these features automatically

API keys are fully encapsulated inside the LLM layer. External callers
never need to handle, pass, or even know about API keys.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator, Iterator, Optional, Any
from enum import Enum
import logging

from shared.key_rotation import KeyRotationMixin, AllKeysFailedError  # noqa: F401

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    GEMINI = "gemini"
    GROQ = "groq"
    CEREBRAS = "cerebras"
    COHERE = "cohere"


@dataclass
class LLMConfig:
    """
    Configuration for LLM instances with built-in resilience.
    
    API keys are optional here — when omitted, the LLM subclass will
    automatically load them from its designated environment variable.
    
    Attributes:
        model: The model identifier (e.g., 'llama-3.3-70b-versatile')
        api_keys: Optional list of API keys (auto-loaded from env if omitted)
        max_retries: Maximum retry attempts on empty response (default: 2)
        temperature: Sampling temperature (0.0 to 2.0)
        max_tokens: Maximum tokens in the response
        top_p: Nucleus sampling parameter
        frequency_penalty: Penalty for token frequency
        presence_penalty: Penalty for token presence
        timeout: Request timeout in seconds
        base_url: Optional custom base URL for API calls
        extra_params: Additional provider-specific parameters
    """
    model: str
    api_keys: Optional[list[str]] = None  # Auto-loaded from env if omitted
    max_retries: int = 2  # Retry attempts on empty response
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    timeout: int = 60
    base_url: Optional[str] = None
    extra_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """
    Standardized response from LLM calls.
    
    Attributes:
        content: The generated text content
        model: The model that generated the response
        provider: The provider that generated the response (e.g., 'groq', 'cerebras')
        finish_reason: Why the generation stopped (e.g., 'stop', 'length')
        usage: Token usage information
        raw_response: The original response from the provider
    """
    content: str
    model: str
    provider: Optional[str] = None
    finish_reason: Optional[str] = None
    usage: Optional[dict[str, int]] = None
    raw_response: Optional[Any] = None
    
    @property
    def prompt_tokens(self) -> int:
        """Get the number of prompt tokens used."""
        if self.usage:
            return self.usage.get("prompt_tokens", 0)
        return 0
    
    @property
    def completion_tokens(self) -> int:
        """Get the number of completion tokens used."""
        if self.usage:
            return self.usage.get("completion_tokens", 0)
        return 0
    
    @property
    def total_tokens(self) -> int:
        """Get the total number of tokens used."""
        if self.usage:
            return self.usage.get("total_tokens", 0)
        return self.prompt_tokens + self.completion_tokens


class BaseLLM(KeyRotationMixin, ABC):
    """
    Abstract base class for all LLM implementations with built-in resilience.
    
    Key rotation, retry, and client caching are provided by
    ``KeyRotationMixin`` from the shared package.  API keys are fully
    encapsulated — they are loaded automatically from environment variables.
    
    Subclasses must define:
    - ENV_VAR_NAME: Class-level constant for the env var name (e.g., "GROQ_API_KEYS")
    - _initialize_client(api_key): Set up the provider client
    - _do_chat(session): Raw chat call
    - _do_chat_async(session): Raw async chat call
    - _do_chat_stream(session): Raw streaming call
    - _do_chat_stream_async(session): Raw async streaming call
    
    Usage:
        class MyLLM(BaseLLM):
            ENV_VAR_NAME = "MY_API_KEYS"
            def _initialize_client(self, api_key):
                self._client = SomeSDK(api_key=api_key)
            def _do_chat(self, session):
                return self._client.chat(...)
    """
    
    ENV_VAR_NAME: str = ""  # Subclasses must override with their env var name
    
    def __init__(self, config: LLMConfig):
        """
        Initialize the LLM with the given configuration.
        
        If config.api_keys is not provided, keys are automatically loaded
        from the environment variable specified by ENV_VAR_NAME.
        
        Args:
            config: LLMConfig instance with model settings
        """
        self.config = config
        self._init_key_rotation(config.api_keys, config.max_retries)
    
    @abstractmethod
    def _initialize_client(self, api_key: str) -> None:
        """
        Initialize the provider-specific client with given API key.
        
        Must store the client as ``self._client`` (and optionally
        ``self._async_client``).  The mixin caches both automatically.
        
        Args:
            api_key: The API key to use for this client
        """
        pass
    
    @property
    @abstractmethod
    def provider(self) -> LLMProvider:
        """Return the provider type for this LLM."""
        pass
    
    def _is_response_empty(self, response: "LLMResponse") -> bool:
        """Check if response content is empty or just whitespace."""
        if not response or not response.content:
            return True
        return response.content.strip() == ""
    
    # ==================== Abstract methods for raw API calls ====================
    
    @abstractmethod
    def _do_chat(self, session: "ChatSession") -> "LLMResponse":
        """
        Raw chat call without resilience logic.
        Subclasses implement the actual API call here.
        """
        pass
    
    @abstractmethod
    async def _do_chat_async(self, session: "ChatSession") -> "LLMResponse":
        """
        Raw async chat call without resilience logic.
        Subclasses implement the actual API call here.
        """
        pass
    
    @abstractmethod
    def _do_chat_stream(self, session: "ChatSession") -> Iterator[str]:
        """
        Raw streaming call without resilience logic.
        Subclasses implement the actual API call here.
        """
        pass
    
    @abstractmethod
    async def _do_chat_stream_async(self, session: "ChatSession") -> AsyncIterator[str]:
        """
        Raw async streaming call without resilience logic.
        Subclasses implement the actual API call here.
        """
        pass
    
    # ==================== Public methods with resilience ====================
    
    def chat(self, session: "ChatSession") -> "LLMResponse":
        """
        Send a chat request with automatic key rotation and retry.
        
        Tries all API keys on failure. Retries on empty response.
        
        Args:
            session: ChatSession containing the conversation history
            
        Returns:
            LLMResponse with the generated content
            
        Raises:
            AllKeysFailedError: If all API keys fail after all retries
        """
        return self._execute_with_rotation(
            operation=lambda: self._do_chat(session),
            is_valid=lambda r: not self._is_response_empty(r),
            service_label=self.provider.value,
            model_label=self.config.model,
        )
    
    async def chat_async(self, session: "ChatSession") -> "LLMResponse":
        """
        Async chat request with automatic key rotation and retry.
        
        Tries all API keys on failure. Retries on empty response.
        
        Args:
            session: ChatSession containing the conversation history
            
        Returns:
            LLMResponse with the generated content
            
        Raises:
            AllKeysFailedError: If all API keys fail after all retries
        """
        return await self._execute_with_rotation_async(
            operation=lambda: self._do_chat_async(session),
            is_valid=lambda r: not self._is_response_empty(r),
            service_label=self.provider.value,
            model_label=self.config.model,
        )
    
    def chat_stream(self, session: "ChatSession") -> Iterator[str]:
        """
        Stream chat with automatic key rotation on failure.
        
        Args:
            session: ChatSession containing the conversation history
            
        Yields:
            String chunks of the response as they arrive
            
        Raises:
            AllKeysFailedError: If all API keys fail
        """
        yield from self._stream_with_rotation(
            operation=lambda: self._do_chat_stream(session),
            service_label=self.provider.value,
            model_label=self.config.model,
        )
    
    async def chat_stream_async(self, session: "ChatSession") -> AsyncIterator[str]:
        """
        Async stream chat with automatic key rotation on failure.
        
        Args:
            session: ChatSession containing the conversation history
            
        Yields:
            String chunks of the response as they arrive
            
        Raises:
            AllKeysFailedError: If all API keys fail
        """
        async for chunk in self._stream_with_rotation_async(
            operation=lambda: self._do_chat_stream_async(session),
            service_label=self.provider.value,
            model_label=self.config.model,
        ):
            yield chunk
    
    def _format_messages(self, session: "ChatSession") -> list[dict[str, str]]:
        """
        Format the session messages for API consumption.
        
        Args:
            session: ChatSession containing the conversation history
            
        Returns:
            List of message dictionaries formatted for the API
        """
        return session.to_api_format()
    
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text string.
        
        This is a basic implementation that can be overridden
        by specific providers for more accurate counting.
        
        Args:
            text: The text to count tokens for
            
        Returns:
            Estimated token count
        """
        # Basic estimation: ~4 characters per token
        return len(text) // 4
    
    def validate_config(self) -> bool:
        """
        Validate the LLM configuration.
        
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        if not self._api_keys:
            raise ValueError("API keys are required")
        if not self.config.model:
            raise ValueError("Model name is required")
        if self.config.temperature < 0 or self.config.temperature > 2:
            raise ValueError("Temperature must be between 0 and 2")
        if self.config.max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        return True
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.config.model})"


# Import here to avoid circular imports
from ..session.chat_session import ChatSession
