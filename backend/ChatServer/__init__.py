"""
ChatServer - A modular LLM chat abstraction layer.

This package provides a generic interface for interacting with various LLM providers
(Groq, Cerebras) through a unified API.
"""

from .llm.base import BaseLLM, LLMConfig, LLMResponse, LLMProvider
from .llm.proprietary_llms.groq_llm import GroqLLM
from .llm.proprietary_llms.cerebras_llm import CerebrasLLM
from .llm.factory import LLMFactory
from .llm.model_registry import (
    MODEL_REGISTRY,
    get_provider_for_model,
    get_all_models_for_provider,
    is_model_supported,
    get_all_supported_models,
    get_provider_model_count,
)
from .session.chat_session import ChatSession, Message, MessageRole
from .services.chat_service import (
    ChatService,
    ProviderConfig,
    AllProvidersFailedError,
    AllKeysFailedError,
    get_chat_service,
    chat_inference,
    chat_inference_stream,
    chat_inference_async,
    chat_inference_stream_async,
)

__all__ = [
    # Base classes
    "BaseLLM",
    "LLMConfig",
    "LLMResponse",
    "LLMProvider",
    # LLM implementations
    "GroqLLM",
    "CerebrasLLM",
    "LLMFactory",
    # Model registry
    "MODEL_REGISTRY",
    "get_provider_for_model",
    "get_all_models_for_provider",
    "is_model_supported",
    "get_all_supported_models",
    "get_provider_model_count",
    # Session management
    "ChatSession",
    "Message",
    "MessageRole",
    # Chat service (for FastAPI)
    "ChatService",
    "ProviderConfig",
    "AllProvidersFailedError",
    "AllKeysFailedError",
    "get_chat_service",
    "chat_inference",
    "chat_inference_stream",
    "chat_inference_async",
    "chat_inference_stream_async",
]
