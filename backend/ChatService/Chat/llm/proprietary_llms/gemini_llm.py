"""
Google Gemini LLM implementation using LangChain.

Provides integration with Google's Gemini models via LangChain.
"""

from typing import AsyncIterator, Optional
from ..base import BaseLLM, LLMConfig, LLMResponse, LLMProvider
from ...session.chat_session import ChatSession, MessageRole


class GeminiLLM(BaseLLM):
    """
    Google Gemini LLM implementation using LangChain.
    
    Supports models like:
    - gemini-1.5-pro
    - gemini-1.5-flash
    - gemini-2.0-flash
    - gemini-pro
    
    Example:
        config = LLMConfig(
            model="gemini-1.5-pro",
            api_key="your-api-key",
            temperature=0.7
        )
        llm = GeminiLLM(config)
        response = llm.chat(session)
    """
    
    def _initialize_client(self) -> None:
        """Initialize the LangChain ChatGoogleGenerativeAI client."""
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            
            self._client = ChatGoogleGenerativeAI(
                model=self.config.model,
                google_api_key=self.config.api_key,
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                **self.config.extra_params,
            )
        except ImportError:
            raise ImportError(
                "LangChain Google GenAI package is required. "
                "Install it with: pip install langchain-google-genai"
            )
    
    @property
    def provider(self) -> LLMProvider:
        """Return the provider type."""
        return LLMProvider.GEMINI
    
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
            if "usage_metadata" in metadata:
                usage_meta = metadata["usage_metadata"]
                usage = {
                    "prompt_tokens": usage_meta.get("prompt_token_count", 0),
                    "completion_tokens": usage_meta.get("candidates_token_count", 0),
                    "total_tokens": usage_meta.get("total_token_count", 0),
                }
            elif "token_usage" in metadata:
                token_usage = metadata["token_usage"]
                usage = {
                    "prompt_tokens": token_usage.get("prompt_tokens", 0),
                    "completion_tokens": token_usage.get("completion_tokens", 0),
                    "total_tokens": token_usage.get("total_tokens", 0),
                }
        return usage
    
    def chat(self, session: ChatSession) -> LLMResponse:
        """
        Send a chat request to Gemini via LangChain.
        
        Args:
            session: ChatSession with conversation history
            
        Returns:
            LLMResponse with the generated content
        """
        messages = self._prepare_messages(session)
        response = self._client.invoke(messages)
        
        return LLMResponse(
            content=response.content,
            model=self.config.model,
            finish_reason=response.response_metadata.get("finish_reason"),
            usage=self._extract_usage(response),
            raw_response=response,
        )
    
    async def chat_async(self, session: ChatSession) -> LLMResponse:
        """
        Asynchronously send a chat request to Gemini via LangChain.
        
        Args:
            session: ChatSession with conversation history
            
        Returns:
            LLMResponse with the generated content
        """
        messages = self._prepare_messages(session)
        response = await self._client.ainvoke(messages)
        
        return LLMResponse(
            content=response.content,
            model=self.config.model,
            finish_reason=response.response_metadata.get("finish_reason"),
            usage=self._extract_usage(response),
            raw_response=response,
        )
    
    def chat_stream(self, session: ChatSession):
        """
        Stream chat responses from Gemini via LangChain.
        
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
        Asynchronously stream chat responses from Gemini via LangChain.
        
        Args:
            session: ChatSession with conversation history
            
        Yields:
            String chunks of the response
        """
        messages = self._prepare_messages(session)
        
        async for chunk in self._client.astream(messages):
            if chunk.content:
                yield chunk.content

