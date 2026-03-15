"""
Cohere LLM implementation using the direct Cohere SDK.

Provides integration with Cohere's Command models via the cohere Python SDK.
Inherits resilience features (key rotation, retry) from BaseLLM.

Uses cohere.ClientV2 / AsyncClientV2 directly (not LangChain) for full
feature support including grounded generation and citations.
"""

from typing import AsyncIterator, Iterator, Optional, Any
from ..base import BaseLLM, LLMConfig, LLMResponse, LLMProvider
from ...session.chat_session import ChatSession, MessageRole
import logging
import os

logger = logging.getLogger(__name__)


class CohereLLM(BaseLLM):
    """
    Cohere LLM implementation using the direct Cohere SDK.
    
    API keys are loaded automatically from the COHERE_API_KEYS environment
    variable (falls back to COHERE_API_KEY for backward compat).
    Resilience features (key rotation, retry on empty) are
    inherited from BaseLLM via KeyRotationMixin.
    
    Supports models like:
    - command-a-03-2025 (latest, most capable)
    - command-r-08-2024
    - command-r-plus-08-2024
    - command-r7b-12-2024 (lightweight)
    
    Example:
        config = LLMConfig(model="command-a-03-2025")
        llm = CohereLLM(config)  # Keys loaded from COHERE_API_KEYS env var
        response = llm.chat(session)  # Automatic key rotation on failure
    """
    
    ENV_VAR_NAME = "COHERE_API_KEYS"
    _FALLBACK_ENV_VAR = "COHERE_API_KEY"
    
    def _load_api_keys_from_env(self) -> list[str]:
        """Try COHERE_API_KEYS first, fall back to COHERE_API_KEY."""
        keys = super()._load_api_keys_from_env()
        if not keys:
            single = os.getenv(self._FALLBACK_ENV_VAR, "").strip()
            if single:
                keys = [single]
        return keys
    
    def _initialize_client(self, api_key: str) -> None:
        """Initialize the Cohere ClientV2 with given API key."""
        try:
            from cohere import ClientV2, AsyncClientV2
            
            self._client = ClientV2(
                api_key=api_key,
                timeout=self.config.timeout,
            )
            self._async_client = AsyncClientV2(
                api_key=api_key,
                timeout=self.config.timeout,
            )
        except ImportError:
            raise ImportError(
                "Cohere package is required. "
                "Install it with: pip install cohere>=5.0.0"
            )
    
    @property
    def provider(self) -> LLMProvider:
        """Return the provider type."""
        return LLMProvider.COHERE
    
    def _prepare_messages(self, session: ChatSession) -> list[dict[str, str]]:
        """
        Convert ChatSession messages to Cohere message format.
        
        Cohere V2 Chat API expects messages as:
        [{"role": "system"|"user"|"assistant", "content": "..."}]
        
        Args:
            session: ChatSession with conversation history
            
        Returns:
            List of message dictionaries for Cohere API
        """
        messages = []
        
        # Add system prompt if present
        if session.system_prompt:
            messages.append({
                "role": "system",
                "content": session.system_prompt,
            })
        
        # Add conversation messages
        for msg in session.messages:
            if msg.role == MessageRole.USER:
                messages.append({"role": "user", "content": msg.content})
            elif msg.role == MessageRole.ASSISTANT:
                messages.append({"role": "assistant", "content": msg.content})
            elif msg.role == MessageRole.SYSTEM:
                messages.append({"role": "system", "content": msg.content})
        
        return messages
    
    def _extract_usage(self, response) -> dict[str, int]:
        """Extract token usage from Cohere response."""
        usage = {}
        if hasattr(response, "usage") and response.usage:
            tokens = response.usage.tokens
            if tokens:
                usage = {
                    "prompt_tokens": getattr(tokens, "input_tokens", 0) or 0,
                    "completion_tokens": getattr(tokens, "output_tokens", 0) or 0,
                    "total_tokens": (
                        (getattr(tokens, "input_tokens", 0) or 0) +
                        (getattr(tokens, "output_tokens", 0) or 0)
                    ),
                }
        return usage
    
    def _extract_content(self, response) -> str:
        """Extract text content from Cohere response."""
        if hasattr(response, "message") and response.message:
            content_parts = response.message.content
            if content_parts and len(content_parts) > 0:
                return content_parts[0].text
        return ""
    
    def _extract_finish_reason(self, response) -> Optional[str]:
        """Extract finish reason from Cohere response."""
        if hasattr(response, "finish_reason"):
            return str(response.finish_reason)
        return None
    
    def _do_chat(self, session: ChatSession) -> LLMResponse:
        """
        Raw chat call to Cohere via direct SDK.
        Resilience logic is handled by BaseLLM.chat()
        
        Args:
            session: ChatSession with conversation history
            
        Returns:
            LLMResponse with the generated content
        """
        messages = self._prepare_messages(session)
        
        kwargs = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        
        # Add any extra params
        if self.config.extra_params:
            kwargs.update(self.config.extra_params)
        
        response = self._client.chat(**kwargs)
        
        return LLMResponse(
            content=self._extract_content(response),
            model=self.config.model,
            provider=self.provider.value,
            finish_reason=self._extract_finish_reason(response),
            usage=self._extract_usage(response),
            raw_response=response,
        )
    
    async def _do_chat_async(self, session: ChatSession) -> LLMResponse:
        """
        Raw async chat call to Cohere via direct SDK.
        Resilience logic is handled by BaseLLM.chat_async()
        
        Args:
            session: ChatSession with conversation history
            
        Returns:
            LLMResponse with the generated content
        """
        messages = self._prepare_messages(session)
        
        kwargs = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        
        if self.config.extra_params:
            kwargs.update(self.config.extra_params)
        
        response = await self._async_client.chat(**kwargs)
        
        return LLMResponse(
            content=self._extract_content(response),
            model=self.config.model,
            provider=self.provider.value,
            finish_reason=self._extract_finish_reason(response),
            usage=self._extract_usage(response),
            raw_response=response,
        )
    
    def _do_chat_stream(self, session: ChatSession) -> Iterator[str]:
        """
        Raw streaming call to Cohere via direct SDK.
        Resilience logic is handled by BaseLLM.chat_stream()
        
        Args:
            session: ChatSession with conversation history
            
        Yields:
            String chunks of the response
        """
        messages = self._prepare_messages(session)
        
        kwargs = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        
        if self.config.extra_params:
            kwargs.update(self.config.extra_params)
        
        stream = self._client.chat_stream(**kwargs)
        
        for event in stream:
            if event.type == "content-delta":
                delta = event.delta
                if delta and hasattr(delta, "message") and delta.message:
                    content = delta.message.content
                    if content and hasattr(content, "text") and content.text:
                        yield content.text
    
    async def _do_chat_stream_async(self, session: ChatSession) -> AsyncIterator[str]:
        """
        Raw async streaming call to Cohere via direct SDK.
        Resilience logic is handled by BaseLLM.chat_stream_async()
        
        Args:
            session: ChatSession with conversation history
            
        Yields:
            String chunks of the response
        """
        messages = self._prepare_messages(session)
        
        kwargs = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        
        if self.config.extra_params:
            kwargs.update(self.config.extra_params)
        
        stream = self._async_client.chat_stream(**kwargs)
        
        async for event in stream:
            if event.type == "content-delta":
                delta = event.delta
                if delta and hasattr(delta, "message") and delta.message:
                    content = delta.message.content
                    if content and hasattr(content, "text") and content.text:
                        yield content.text
