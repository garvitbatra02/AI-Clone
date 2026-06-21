"""
Mistral AI LLM implementation using LangChain.

Provides integration with Mistral AI's API via LangChain.
Inherits resilience features (key rotation, retry) from BaseLLM.
"""

from typing import AsyncIterator, Iterator
from ..base import BaseLLM, LLMConfig, LLMResponse, LLMProvider
from ...session.chat_session import ChatSession, MessageRole


class MistralLLM(BaseLLM):
    """
    Mistral AI LLM implementation using LangChain.
    
    API keys are loaded automatically from the MISTRAL_API_KEYS environment
    variable. Mistral provides frontier-class models for chat, code,
    and reasoning via an OpenAI-compatible API at api.mistral.ai.
    
    Resilience features (key rotation, retry on empty) are inherited from BaseLLM.
    
    Supports models like:
    - mistral-large-latest (flagship)
    - mistral-medium-latest (balanced)
    - mistral-small-latest (efficient)
    - codestral-latest (code generation)
    - magistral-medium-latest (reasoning)
    - ministral-8b-latest (edge/lightweight)
    - open-mistral-nemo (open-weight)
    
    Example:
        config = LLMConfig(model="mistral-small-latest")
        llm = MistralLLM(config)  # Keys loaded from MISTRAL_API_KEYS env var
        response = llm.chat(session)  # Automatic key rotation on failure
    """
    
    ENV_VAR_NAME = "MISTRAL_API_KEYS"
    
    def _initialize_client(self, api_key: str) -> None:
        """Initialize the LangChain ChatMistralAI client with given API key."""
        try:
            from langchain_mistralai import ChatMistralAI
            
            self._client = ChatMistralAI(
                model=self.config.model,
                api_key=api_key,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                **self.config.extra_params,
            )
        except ImportError:
            raise ImportError(
                "LangChain Mistral AI package is required. "
                "Install it with: pip install langchain-mistralai"
            )
    
    @property
    def provider(self) -> LLMProvider:
        """Return the provider type."""
        return LLMProvider.MISTRAL
    
    def _prepare_messages(self, session: ChatSession) -> list:
        """
        Convert ChatSession messages to LangChain message format.
        
        Args:
            session: ChatSession with conversation history
            
        Returns:
            List of LangChain message objects
        """
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
        
        messages = []
        
        # Add system prompt if present
        if session.system_prompt:
            messages.append(SystemMessage(content=session.system_prompt))
        
        # Add conversation messages
        for msg in session.messages:
            if msg.role == MessageRole.USER:
                messages.append(HumanMessage(content=msg.content))
            elif msg.role == MessageRole.ASSISTANT:
                messages.append(AIMessage(content=msg.content))
            elif msg.role == MessageRole.SYSTEM:
                messages.append(SystemMessage(content=msg.content))
        
        return messages
    
    def _extract_usage(self, response) -> dict:
        """Extract token usage from LangChain response."""
        usage = {}
        if hasattr(response, "response_metadata"):
            metadata = response.response_metadata
            if "token_usage" in metadata:
                token_usage = metadata["token_usage"]
                usage = {
                    "prompt_tokens": token_usage.get("prompt_tokens", 0),
                    "completion_tokens": token_usage.get("completion_tokens", 0),
                    "total_tokens": token_usage.get("total_tokens", 0),
                }
            elif "usage" in metadata:
                usage_data = metadata["usage"]
                usage = {
                    "prompt_tokens": usage_data.get("prompt_tokens", 0),
                    "completion_tokens": usage_data.get("completion_tokens", 0),
                    "total_tokens": usage_data.get("total_tokens", 0),
                }
        return usage
    
    def _do_chat(self, session: ChatSession) -> LLMResponse:
        """
        Raw chat call to Mistral AI via LangChain.
        Resilience logic is handled by BaseLLM.chat()
        """
        messages = self._prepare_messages(session)
        response = self._client.invoke(messages)
        
        finish_reason = None
        if hasattr(response, "response_metadata"):
            finish_reason = response.response_metadata.get("finish_reason")
        
        return LLMResponse(
            content=response.content,
            model=self.config.model,
            provider=self.provider.value,
            finish_reason=finish_reason,
            usage=self._extract_usage(response),
            raw_response=response,
        )
    
    async def _do_chat_async(self, session: ChatSession) -> LLMResponse:
        """
        Raw async chat call to Mistral AI via LangChain.
        Resilience logic is handled by BaseLLM.chat_async()
        """
        messages = self._prepare_messages(session)
        response = await self._client.ainvoke(messages)
        
        finish_reason = None
        if hasattr(response, "response_metadata"):
            finish_reason = response.response_metadata.get("finish_reason")
        
        return LLMResponse(
            content=response.content,
            model=self.config.model,
            provider=self.provider.value,
            finish_reason=finish_reason,
            usage=self._extract_usage(response),
            raw_response=response,
        )
    
    def _do_chat_stream(self, session: ChatSession) -> Iterator[str]:
        """
        Raw streaming call to Mistral AI via LangChain.
        Resilience logic is handled by BaseLLM.chat_stream()
        """
        messages = self._prepare_messages(session)
        
        for chunk in self._client.stream(messages):
            if chunk.content:
                yield chunk.content
    
    async def _do_chat_stream_async(self, session: ChatSession, *, stats=None) -> AsyncIterator[str]:
        """
        Raw async streaming call to Mistral AI via LangChain.
        Resilience logic is handled by BaseLLM.chat_stream_async()
        """
        from ..base import normalize_langchain_usage
        messages = self._prepare_messages(session)
        
        async for chunk in self._client.astream(messages):
            if stats is not None:
                usage = normalize_langchain_usage(chunk)
                if usage:
                    stats["usage"] = usage
            if chunk.content:
                yield chunk.content
