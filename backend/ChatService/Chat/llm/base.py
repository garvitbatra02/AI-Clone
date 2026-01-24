"""
Base LLM class providing the abstract interface for all LLM providers.

This module defines the contract that all LLM implementations must follow,
ensuring a consistent API across different providers.

Resilience features are built into BaseLLM:
- Multiple API keys with automatic rotation
- Retry on failure or empty response
- All LLM implementations inherit these features automatically
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator, Iterator, Optional, Any, Union
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    GEMINI = "gemini"
    GROQ = "groq"
    CEREBRAS = "cerebras"


class AllKeysFailedError(Exception):
    """
    Raised when all API keys for an LLM fail.
    
    Attributes:
        provider: The provider that was being used.
        model: The model that was requested.
        errors: Dictionary mapping key indices to their error messages.
    """
    def __init__(self, provider: str, model: str, errors: dict[int, str]):
        self.provider = provider
        self.model = model
        self.errors = errors
        key_errors = "; ".join([f"key_{k}: {v}" for k, v in errors.items()])
        super().__init__(
            f"All API keys failed for {provider} with model {model}. Errors: {key_errors}"
        )


@dataclass
class LLMConfig:
    """
    Configuration for LLM instances with built-in resilience.
    
    Attributes:
        model: The model identifier (e.g., 'gemini-1.5-pro', 'llama-3.1-70b')
        api_keys: List of API keys for rotation and fallback
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
    api_keys: list[str]  # Multiple API keys for resilience
    max_retries: int = 2  # Retry attempts on empty response
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    timeout: int = 60
    base_url: Optional[str] = None
    extra_params: dict[str, Any] = field(default_factory=dict)
    
    @property
    def api_key(self) -> str:
        """Return first API key for backward compatibility."""
        return self.api_keys[0] if self.api_keys else ""
    
    @classmethod
    def from_single_key(cls, model: str, api_key: str, **kwargs) -> "LLMConfig":
        """Create config from a single API key (backward compatibility)."""
        return cls(model=model, api_keys=[api_key], **kwargs)


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


class BaseLLM(ABC):
    """
    Abstract base class for all LLM implementations with built-in resilience.
    
    This class provides:
    - Automatic API key rotation on failure
    - Retry logic for empty responses
    - Consistent error handling
    
    Subclasses only need to implement:
    - _initialize_client(api_key): Set up the provider client
    - _do_chat(session): Raw chat call
    - _do_chat_async(session): Raw async chat call
    - _do_chat_stream(session): Raw streaming call
    - _do_chat_stream_async(session): Raw async streaming call
    
    Usage:
        class MyLLM(BaseLLM):
            def _do_chat(self, session):
                # Raw API call implementation
                pass
    """
    
    def __init__(self, config: LLMConfig):
        """
        Initialize the LLM with the given configuration.
        
        Args:
            config: LLMConfig instance with model settings and API keys
        """
        self.config = config
        self._clients: dict[int, Any] = {}  # key_index -> client
        self._current_key_index = 0
        
        # Initialize client with first key
        self._initialize_client(self.config.api_keys[0])
    
    @abstractmethod
    def _initialize_client(self, api_key: str) -> None:
        """
        Initialize the provider-specific client with given API key.
        
        Args:
            api_key: The API key to use for this client
        """
        pass
    
    def _get_client_for_key(self, key_index: int) -> Any:
        """
        Get or create a client for a specific API key.
        
        Args:
            key_index: Index of the API key to use
            
        Returns:
            Provider client instance
        """
        if key_index not in self._clients:
            api_key = self.config.api_keys[key_index]
            # Store current client
            old_client = getattr(self, '_client', None)
            # Initialize new client
            self._initialize_client(api_key)
            # Cache it
            self._clients[key_index] = self._client
            # Restore if needed
            if old_client and key_index != 0:
                pass  # Keep the new client as current
        return self._clients[key_index]
    
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
        for retry_attempt in range(self.config.max_retries + 1):
            errors: dict[int, str] = {}
            
            if retry_attempt > 0:
                logger.info(
                    f"Retrying all keys for {self.provider.value} "
                    f"(attempt {retry_attempt + 1}/{self.config.max_retries + 1})..."
                )
            
            for key_index, api_key in enumerate(self.config.api_keys):
                try:
                    # Get client for this key
                    self._client = self._get_client_for_key(key_index)
                    
                    response = self._do_chat(session)
                    
                    # Validate response is not empty
                    if self._is_response_empty(response):
                        logger.warning(
                            f"{self.provider.value} key {key_index} returned empty response "
                            f"(attempt {retry_attempt + 1}). Trying next key..."
                        )
                        errors[key_index] = "Empty response"
                        continue
                    
                    # Success
                    logger.info(
                        f"Successfully served request using {self.provider.value} "
                        f"key {key_index} (content length: {len(response.content)})"
                    )
                    return response
                    
                except Exception as e:
                    error_msg = str(e)
                    errors[key_index] = error_msg
                    logger.warning(
                        f"{self.provider.value} key {key_index} failed: {error_msg}. "
                        f"Trying next key..."
                    )
            
            # All keys failed for this attempt
            if retry_attempt < self.config.max_retries:
                logger.warning(
                    f"All keys for {self.provider.value} failed. Retrying..."
                )
        
        # All retries exhausted
        logger.error(
            f"All {len(self.config.api_keys)} keys failed for {self.provider.value} "
            f"after {self.config.max_retries + 1} attempts"
        )
        raise AllKeysFailedError(self.provider.value, self.config.model, errors)
    
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
        for retry_attempt in range(self.config.max_retries + 1):
            errors: dict[int, str] = {}
            
            if retry_attempt > 0:
                logger.info(
                    f"Retrying all async keys for {self.provider.value} "
                    f"(attempt {retry_attempt + 1}/{self.config.max_retries + 1})..."
                )
            
            for key_index, api_key in enumerate(self.config.api_keys):
                try:
                    # Get client for this key
                    self._client = self._get_client_for_key(key_index)
                    
                    response = await self._do_chat_async(session)
                    
                    # Validate response is not empty
                    if self._is_response_empty(response):
                        logger.warning(
                            f"{self.provider.value} key {key_index} returned empty response "
                            f"(attempt {retry_attempt + 1}). Trying next key..."
                        )
                        errors[key_index] = "Empty response"
                        continue
                    
                    # Success
                    logger.info(
                        f"Successfully served async request using {self.provider.value} "
                        f"key {key_index} (content length: {len(response.content)})"
                    )
                    return response
                    
                except Exception as e:
                    error_msg = str(e)
                    errors[key_index] = error_msg
                    logger.warning(
                        f"{self.provider.value} key {key_index} failed: {error_msg}. "
                        f"Trying next key..."
                    )
            
            # All keys failed for this attempt
            if retry_attempt < self.config.max_retries:
                logger.warning(
                    f"All async keys for {self.provider.value} failed. Retrying..."
                )
        
        # All retries exhausted
        logger.error(
            f"All {len(self.config.api_keys)} async keys failed for {self.provider.value} "
            f"after {self.config.max_retries + 1} attempts"
        )
        raise AllKeysFailedError(self.provider.value, self.config.model, errors)
    
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
        errors: dict[int, str] = {}
        
        for key_index, api_key in enumerate(self.config.api_keys):
            try:
                # Get client for this key
                self._client = self._get_client_for_key(key_index)
                
                # Try to get first chunk to verify connection
                stream = self._do_chat_stream(session)
                first_chunk = next(stream)
                
                logger.info(
                    f"Successfully streaming using {self.provider.value} key {key_index}"
                )
                yield first_chunk
                yield from stream
                return
                
            except Exception as e:
                error_msg = str(e)
                errors[key_index] = error_msg
                logger.warning(
                    f"{self.provider.value} key {key_index} failed: {error_msg}. "
                    f"Trying next key..."
                )
        
        # All keys failed
        logger.error(
            f"All {len(self.config.api_keys)} keys failed for streaming {self.provider.value}"
        )
        raise AllKeysFailedError(self.provider.value, self.config.model, errors)
    
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
        errors: dict[int, str] = {}
        
        for key_index, api_key in enumerate(self.config.api_keys):
            try:
                # Get client for this key
                self._client = self._get_client_for_key(key_index)
                
                # Try to get first chunk to verify connection
                stream = self._do_chat_stream_async(session)
                first_chunk = await stream.__anext__()
                
                logger.info(
                    f"Successfully streaming async using {self.provider.value} key {key_index}"
                )
                yield first_chunk
                async for chunk in stream:
                    yield chunk
                return
                
            except Exception as e:
                error_msg = str(e)
                errors[key_index] = error_msg
                logger.warning(
                    f"{self.provider.value} key {key_index} failed: {error_msg}. "
                    f"Trying next key..."
                )
        
        # All keys failed
        logger.error(
            f"All {len(self.config.api_keys)} async keys failed for streaming {self.provider.value}"
        )
        raise AllKeysFailedError(self.provider.value, self.config.model, errors)
    
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
        if not self.config.api_key:
            raise ValueError("API key is required")
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
