"""
Cerebras LLM implementation using LangChain.

Provides integration with Cerebras AI compute platform via LangChain.
Inherits resilience features (key rotation, retry) from BaseLLM.
"""

from typing import AsyncIterator, Iterator
from ..base import BaseLLM, LLMConfig, LLMResponse, LLMProvider
from ...session.chat_session import ChatSession, MessageRole


class CerebrasLLM(BaseLLM):
    """
    Cerebras LLM implementation using LangChain.
    
    Cerebras provides ultra-fast inference with specialized AI processors.
    Resilience features (key rotation, retry on empty) are inherited from BaseLLM.
    This class only implements the raw API calls.
    
    Supports models like:
    - llama3.1-8b
    - llama3.1-70b
    
    Example:
        config = LLMConfig(
            model="llama3.1-70b",
            api_keys=["key1", "key2", "key3"],  # Multiple keys for resilience
            temperature=0.7
        )
        llm = CerebrasLLM(config)
        response = llm.chat(session)  # Automatic key rotation on failure
    """
    
    def _initialize_client(self, api_key: str) -> None:
        """Initialize the LangChain ChatCerebras client with given API key."""
        try:
            from langchain_cerebras import ChatCerebras
            
            self._client = ChatCerebras(
                model=self.config.model,
                api_key=api_key,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                **self.config.extra_params,
            )
        except ImportError:
            raise ImportError(
                "LangChain Cerebras package is required. "
                "Install it with: pip install langchain-cerebras"
            )
    
    @property
    def provider(self) -> LLMProvider:
        """Return the provider type."""
        return LLMProvider.CEREBRAS
    
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
        Raw chat call to Cerebras via LangChain.
        Resilience logic is handled by BaseLLM.chat()
        
        Args:
            session: ChatSession with conversation history
            
        Returns:
            LLMResponse with the generated content
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
        Raw async chat call to Cerebras via LangChain.
        Resilience logic is handled by BaseLLM.chat_async()
        
        Args:
            session: ChatSession with conversation history
            
        Returns:
            LLMResponse with the generated content
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
        Raw streaming call to Cerebras via LangChain.
        Resilience logic is handled by BaseLLM.chat_stream()
        
        Args:
            session: ChatSession with conversation history
            
        Yields:
            String chunks of the response
        """
        messages = self._prepare_messages(session)
        
        for chunk in self._client.stream(messages):
            if chunk.content:
                yield chunk.content
    
    async def _do_chat_stream_async(self, session: ChatSession) -> AsyncIterator[str]:
        """
        Raw async streaming call to Cerebras via LangChain.
        Resilience logic is handled by BaseLLM.chat_stream_async()
        
        Args:
            session: ChatSession with conversation history
            
        Yields:
            String chunks of the response
        """
        messages = self._prepare_messages(session)
        
        async for chunk in self._client.astream(messages):
            if chunk.content:
                yield chunk.content
