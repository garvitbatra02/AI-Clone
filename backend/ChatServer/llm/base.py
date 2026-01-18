"""
Base LLM class providing the abstract interface for all LLM providers.

This module defines the contract that all LLM implementations must follow,
ensuring a consistent API across different providers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional, Any
from enum import Enum


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    GEMINI = "gemini"
    GROQ = "groq"
    CEREBRAS = "cerebras"


@dataclass
class LLMConfig:
    """
    Configuration for LLM instances.
    
    Attributes:
        model: The model identifier (e.g., 'gemini-1.5-pro', 'llama-3.1-70b')
        api_key: API key for the provider
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
    api_key: str
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


class BaseLLM(ABC):
    """
    Abstract base class for all LLM implementations.
    
    This class defines the interface that all LLM providers must implement.
    It provides both synchronous and asynchronous methods for chat completion,
    as well as streaming support.
    
    Usage:
        class MyLLM(BaseLLM):
            def chat(self, session):
                # Implementation here
                pass
    """
    
    def __init__(self, config: LLMConfig):
        """
        Initialize the LLM with the given configuration.
        
        Args:
            config: LLMConfig instance with model settings
        """
        self.config = config
        self._client: Optional[Any] = None
        self._async_client: Optional[Any] = None
        self._initialize_client()
    
    @abstractmethod
    def _initialize_client(self) -> None:
        """
        Initialize the provider-specific client.
        
        This method should set up any necessary client instances
        for making API calls.
        """
        pass
    
    @property
    @abstractmethod
    def provider(self) -> LLMProvider:
        """Return the provider type for this LLM."""
        pass
    
    @abstractmethod
    def chat(self, session: "ChatSession") -> LLMResponse:
        """
        Send a chat request to the LLM and get a response.
        
        Args:
            session: ChatSession containing the conversation history
            
        Returns:
            LLMResponse with the generated content
        """
        pass
    
    @abstractmethod
    async def chat_async(self, session: "ChatSession") -> LLMResponse:
        """
        Asynchronously send a chat request to the LLM.
        
        Args:
            session: ChatSession containing the conversation history
            
        Returns:
            LLMResponse with the generated content
        """
        pass
    
    @abstractmethod
    def chat_stream(self, session: "ChatSession") -> AsyncIterator[str]:
        """
        Stream chat responses from the LLM.
        
        Args:
            session: ChatSession containing the conversation history
            
        Yields:
            String chunks of the response as they arrive
        """
        pass
    
    @abstractmethod
    async def chat_stream_async(self, session: "ChatSession") -> AsyncIterator[str]:
        """
        Asynchronously stream chat responses from the LLM.
        
        Args:
            session: ChatSession containing the conversation history
            
        Yields:
            String chunks of the response as they arrive
        """
        pass
    
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
