"""
ChatServer - A modular LLM chat abstraction layer.

This package provides a generic interface for interacting with various LLM providers
(Google Gemini, Groq, Cerebras) through a unified API.
"""

from .llm.base import BaseLLM, LLMConfig, LLMResponse, LLMProvider
from .llm.proprietary_llms.gemini_llm import GeminiLLM
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

__all__ = [
    # Base classes
    "BaseLLM",
    "LLMConfig",
    "LLMResponse",
    "LLMProvider",
    # LLM implementations
    "GeminiLLM",
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
]
