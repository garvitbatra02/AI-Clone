"""
Groq LLM implementation using LangChain.

Provides integration with Groq's fast inference API via LangChain.
"""

from typing import AsyncIterator
from ..base import BaseLLM, LLMConfig, LLMResponse, LLMProvider
from ...session.chat_session import ChatSession, MessageRole


class GroqLLM(BaseLLM):
    """
    Groq LLM implementation using LangChain for fast inference.
    
    Supports models like:
    - llama-3.3-70b-versatile
    - llama-3.1-70b-versatile
    - llama-3.1-8b-instant
    - mixtral-8x7b-32768
    - gemma-7b-it
    
    Example:
        config = LLMConfig(
            model="llama-3.1-70b-versatile",
            api_key="your-api-key",
            temperature=0.7
        )
        llm = GroqLLM(config)
        response = llm.chat(session)
    """
    
    def _initialize_client(self) -> None:
        """Initialize the LangChain ChatGroq client."""
        try:
            from langchain_groq import ChatGroq
            
            self._client = ChatGroq(
                model=self.config.model,
                api_key=self.config.api_key,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                **self.config.extra_params,
            )
        except ImportError:
            raise ImportError(
                "LangChain Groq package is required. "
                "Install it with: pip install langchain-groq"
            )
    
    @property
    def provider(self) -> LLMProvider:
        """Return the provider type."""
        return LLMProvider.GROQ
    
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
    
    def chat(self, session: ChatSession) -> LLMResponse:
        """
        Send a chat request to Groq via LangChain.
        
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
            finish_reason=finish_reason,
            usage=self._extract_usage(response),
            raw_response=response,
        )
    
    async def chat_async(self, session: ChatSession) -> LLMResponse:
        """
        Asynchronously send a chat request to Groq via LangChain.
        
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
            finish_reason=finish_reason,
            usage=self._extract_usage(response),
            raw_response=response,
        )
    
    def chat_stream(self, session: ChatSession):
        """
        Stream chat responses from Groq via LangChain.
        
        Args:
            session: ChatSession with conversation history
            
        Yields:
            String chunks of the response
        """
        messages = self._prepare_messages(session)
        
        for chunk in self._client.stream(messages):
            if chunk.content:
                yield chunk.content
    
    async def chat_stream_async(self, session: ChatSession) -> AsyncIterator[str]:
        """
        Asynchronously stream chat responses from Groq via LangChain.
        
        Args:
            session: ChatSession with conversation history
            
        Yields:
            String chunks of the response
        """
        messages = self._prepare_messages(session)
        
        async for chunk in self._client.astream(messages):
            if chunk.content:
                yield chunk.content
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
