"""
RAG Service - Retrieval Augmented Generation Orchestrator

Ties the retrieval pipeline to the chat pipeline. This is the main
entry point for RAG queries:

1. Retrieves relevant context via RetrievalService (search → rerank → format)
2. Injects context into a ChatSession via system prompt
3. Generates a response via an isolated ChatService (Cohere → Cerebras → Groq)

Uses its own isolated ChatService instance with RAG-specific provider priority,
separate from the regular chat service singleton.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union

from ChatService.Chat.llm.base import LLMProvider, LLMResponse
from ChatService.Chat.services.chat_service import (
    ChatService,
    ProviderConfig,
    AllProvidersFailedError,
)
from ChatService.Chat.session.chat_session import ChatSession, MessageRole
from RAGService.Data.services.retrieval_service import (
    RetrievalConfig,
    RetrievalResult,
    RetrievalService,
    get_retrieval_service,
)
from prompts import RAG_SYSTEM_PROMPT
from RAGService.Data.VectorDB.base import (
    MetadataFilter,
    MetadataFilterGroup,
    SearchResult,
)

logger = logging.getLogger(__name__)


# ==================== Configuration ====================


# Default RAG provider priority: Cohere → Cerebras → Groq
DEFAULT_RAG_PROVIDERS = [
    ProviderConfig(
        provider=LLMProvider.COHERE,
        default_model="command-a-03-2025",
    ),
    ProviderConfig(
        provider=LLMProvider.CEREBRAS,
        default_model="llama3.1-8b",
    ),
    ProviderConfig(
        provider=LLMProvider.GROQ,
        default_model="llama-3.3-70b-versatile",
    ),
]


@dataclass
class RAGConfig:
    """
    Configuration for the RAG Service.
    
    Attributes:
        llm_providers: Ordered list of LLM providers for RAG (priority order)
        temperature: LLM temperature for RAG responses
        max_tokens: Max tokens for LLM response
        retrieval_config: Configuration for the retrieval pipeline
        system_prompt_template: Template for the RAG system prompt.
                               {context} is replaced with the retrieved context.
                               {user_system_prompt} is replaced with any user-provided system prompt.
    """
    llm_providers: List[ProviderConfig] = field(default_factory=lambda: DEFAULT_RAG_PROVIDERS.copy())
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    retrieval_config: Optional[RetrievalConfig] = None
    system_prompt_template: str = RAG_SYSTEM_PROMPT
    
    @classmethod
    def from_env(cls) -> "RAGConfig":
        """Create configuration from environment variables."""
        temperature = float(os.environ.get("RAG_TEMPERATURE", "0.7"))
        max_tokens = os.environ.get("RAG_MAX_TOKENS")
        
        return cls(
            temperature=temperature,
            max_tokens=int(max_tokens) if max_tokens else None,
            retrieval_config=RetrievalConfig.from_env(),
        )


# ==================== Response Models ====================


@dataclass
class RAGResponse:
    """
    Response from a RAG query.
    
    Attributes:
        answer: The LLM-generated answer
        sources: Source chunks used to generate the answer
        provider_used: Which LLM provider served the request
        model_used: Which model generated the response
        retrieval_result: Full retrieval pipeline result
        llm_response: Raw LLM response object
    """
    answer: str
    sources: List[SearchResult]
    provider_used: str
    model_used: str
    retrieval_result: RetrievalResult
    llm_response: Optional[LLMResponse] = None


# ==================== Service ====================


class RAGService:
    """
    RAG orchestrator that combines retrieval with LLM generation.
    
    Manages its own isolated ChatService instance with RAG-specific
    provider priority (Cohere → Cerebras → Groq), completely separate
    from the regular chat service.
    
    Pipeline:
    1. RetrievalService.retrieve() → context string + source chunks
    2. Build ChatSession with context injected into system prompt
    3. ChatService.chat(fallback=True) → LLM response with auto-fallback
    
    Example:
        rag = RAGService()
        response = rag.query(
            user_query="What is machine learning?",
            collection_name="my_docs",
        )
        print(response.answer)
        print(response.sources)
    """
    
    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        retrieval_service: Optional[RetrievalService] = None,
    ):
        """
        Initialize the RAG service.
        
        Creates its own isolated ChatService instance with RAG provider priority.
        
        Args:
            config: RAG configuration (uses defaults/env if omitted)
            retrieval_service: Pre-configured RetrievalService (uses singleton if omitted)
        """
        self.config = config or RAGConfig.from_env()
        self._retrieval_service = retrieval_service
        
        # Create isolated ChatService for RAG with its own provider priority
        self._chat_service = ChatService(
            providers=self.config.llm_providers,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        
        logger.info(
            f"RAG Service initialized with providers: "
            f"{[p.provider.value for p in self.config.llm_providers]}"
        )
    
    @property
    def retrieval_service(self) -> RetrievalService:
        """Get or create the retrieval service (lazy)."""
        if self._retrieval_service is None:
            self._retrieval_service = get_retrieval_service(
                config=self.config.retrieval_config,
            )
        return self._retrieval_service
    
    def _build_rag_session(
        self,
        context_str: str,
        user_query: str,
        session: Optional[ChatSession] = None,
        system_prompt: Optional[str] = None,
        sources: Optional[List[SearchResult]] = None,
    ) -> ChatSession:
        """
        Build a ChatSession with RAG context injected into the system prompt.
        
        Args:
            context_str: Formatted context from retrieval
            user_query: The user's question
            session: Existing session to augment (creates new if None)
            system_prompt: Additional user-provided system prompt
            sources: Source chunks for session context storage
            
        Returns:
            ChatSession ready for LLM generation
        """
        if session is None:
            session = ChatSession()
        
        # Store RAG data in session context for reference
        session.set_context("rag_context", context_str)
        if sources:
            session.set_context("rag_sources", [
                {"id": s.id, "content": s.content[:200], "score": s.score}
                for s in sources
            ])
        
        # Build the system prompt with context
        user_system_prompt = system_prompt or ""
        rag_system_prompt = self.config.system_prompt_template.format(
            context=context_str,
            user_system_prompt=user_system_prompt,
        ).strip()
        
        session.add_system_prompt(rag_system_prompt)
        
        # Add the user query as the latest message if not already in session
        if not session.messages or session.get_last_user_message() is None or \
           session.get_last_user_message().content != user_query:
            session.add_user_message(user_query)
        
        return session
    
    # ==================== Core Query Methods ====================
    
    def query(
        self,
        user_query: str,
        collection_name: Optional[str] = None,
        session: Optional[ChatSession] = None,
        system_prompt: Optional[str] = None,
        filters: Optional[Union[MetadataFilter, MetadataFilterGroup]] = None,
        provider: Optional[LLMProvider] = None,
        model: Optional[str] = None,
        top_k: Optional[int] = None,
        rerank_top_n: Optional[int] = None,
    ) -> RAGResponse:
        """
        Execute a full RAG query: retrieve → inject context → generate.
        
        Args:
            user_query: The user's question
            collection_name: Vector DB collection to search
            session: Existing ChatSession to augment (creates new if None)
            system_prompt: Additional system prompt to include
            filters: Metadata filters for retrieval
            provider: Force a specific LLM provider
            model: Force a specific model
            top_k: Override retrieval top_k
            rerank_top_n: Override reranker top_n
            
        Returns:
            RAGResponse with answer, sources, and metadata
        """
        # Stage 1: Retrieve
        retrieval_result = self.retrieval_service.retrieve(
            query=user_query,
            collection_name=collection_name,
            filters=filters,
            top_k=top_k,
            rerank_top_n=rerank_top_n,
        )
        
        logger.info(
            f"Retrieved {len(retrieval_result.source_chunks)} chunks "
            f"(reranked={retrieval_result.reranked})"
        )
        
        # Stage 2: Build session with context
        rag_session = self._build_rag_session(
            context_str=retrieval_result.context_str,
            user_query=user_query,
            session=session,
            system_prompt=system_prompt,
            sources=retrieval_result.source_chunks,
        )
        
        # Stage 3: Generate with LLM (auto-fallback through provider priority)
        llm_response = self._chat_service.chat(
            session=rag_session,
            provider=provider,
            fallback=True,
            model=model,
        )
        
        logger.info(
            f"RAG response generated by {llm_response.provider}/{llm_response.model}"
        )
        
        return RAGResponse(
            answer=llm_response.content,
            sources=retrieval_result.source_chunks,
            provider_used=llm_response.provider or "unknown",
            model_used=llm_response.model,
            retrieval_result=retrieval_result,
            llm_response=llm_response,
        )
    
    async def aquery(
        self,
        user_query: str,
        collection_name: Optional[str] = None,
        session: Optional[ChatSession] = None,
        system_prompt: Optional[str] = None,
        filters: Optional[Union[MetadataFilter, MetadataFilterGroup]] = None,
        provider: Optional[LLMProvider] = None,
        model: Optional[str] = None,
        top_k: Optional[int] = None,
        rerank_top_n: Optional[int] = None,
    ) -> RAGResponse:
        """
        Async version of query.
        
        Args:
            user_query: The user's question
            collection_name: Vector DB collection to search
            session: Existing ChatSession
            system_prompt: Additional system prompt
            filters: Metadata filters
            provider: Force a specific LLM provider
            model: Force a specific model
            top_k: Override retrieval top_k
            rerank_top_n: Override reranker top_n
            
        Returns:
            RAGResponse with answer, sources, and metadata
        """
        # Stage 1: Async retrieve
        retrieval_result = await self.retrieval_service.aretrieve(
            query=user_query,
            collection_name=collection_name,
            filters=filters,
            top_k=top_k,
            rerank_top_n=rerank_top_n,
        )
        
        logger.info(
            f"Async retrieved {len(retrieval_result.source_chunks)} chunks "
            f"(reranked={retrieval_result.reranked})"
        )
        
        # Stage 2: Build session with context
        rag_session = self._build_rag_session(
            context_str=retrieval_result.context_str,
            user_query=user_query,
            session=session,
            system_prompt=system_prompt,
            sources=retrieval_result.source_chunks,
        )
        
        # Stage 3: Async generate
        llm_response = await self._chat_service.chat_async(
            session=rag_session,
            provider=provider,
            fallback=True,
            model=model,
        )
        
        logger.info(
            f"Async RAG response generated by {llm_response.provider}/{llm_response.model}"
        )
        
        return RAGResponse(
            answer=llm_response.content,
            sources=retrieval_result.source_chunks,
            provider_used=llm_response.provider or "unknown",
            model_used=llm_response.model,
            retrieval_result=retrieval_result,
            llm_response=llm_response,
        )
    
    def query_stream(
        self,
        user_query: str,
        collection_name: Optional[str] = None,
        session: Optional[ChatSession] = None,
        system_prompt: Optional[str] = None,
        filters: Optional[Union[MetadataFilter, MetadataFilterGroup]] = None,
        provider: Optional[LLMProvider] = None,
        model: Optional[str] = None,
        top_k: Optional[int] = None,
        rerank_top_n: Optional[int] = None,
    ) -> tuple[RetrievalResult, Iterator[str]]:
        """
        Execute a RAG query with streaming LLM response.
        
        Returns the retrieval result immediately along with a streaming
        iterator for the LLM response.
        
        Args:
            user_query: The user's question
            collection_name: Vector DB collection to search
            session: Existing ChatSession
            system_prompt: Additional system prompt
            filters: Metadata filters
            provider: Force a specific LLM provider
            model: Force a specific model
            top_k: Override retrieval top_k
            rerank_top_n: Override reranker top_n
            
        Returns:
            Tuple of (RetrievalResult, Iterator[str] for streaming chunks)
        """
        # Stage 1: Retrieve
        retrieval_result = self.retrieval_service.retrieve(
            query=user_query,
            collection_name=collection_name,
            filters=filters,
            top_k=top_k,
            rerank_top_n=rerank_top_n,
        )
        
        # Stage 2: Build session
        rag_session = self._build_rag_session(
            context_str=retrieval_result.context_str,
            user_query=user_query,
            session=session,
            system_prompt=system_prompt,
            sources=retrieval_result.source_chunks,
        )
        
        # Stage 3: Stream generate
        stream = self._chat_service.chat_stream(
            session=rag_session,
            provider=provider,
            fallback=True,
            model=model,
        )
        
        return retrieval_result, stream
    
    async def aquery_stream(
        self,
        user_query: str,
        collection_name: Optional[str] = None,
        session: Optional[ChatSession] = None,
        system_prompt: Optional[str] = None,
        filters: Optional[Union[MetadataFilter, MetadataFilterGroup]] = None,
        provider: Optional[LLMProvider] = None,
        model: Optional[str] = None,
        top_k: Optional[int] = None,
        rerank_top_n: Optional[int] = None,
    ) -> tuple[RetrievalResult, AsyncIterator[str]]:
        """
        Async version of query_stream.
        
        Returns:
            Tuple of (RetrievalResult, AsyncIterator[str] for streaming chunks)
        """
        # Stage 1: Async retrieve
        retrieval_result = await self.retrieval_service.aretrieve(
            query=user_query,
            collection_name=collection_name,
            filters=filters,
            top_k=top_k,
            rerank_top_n=rerank_top_n,
        )
        
        # Stage 2: Build session
        rag_session = self._build_rag_session(
            context_str=retrieval_result.context_str,
            user_query=user_query,
            session=session,
            system_prompt=system_prompt,
            sources=retrieval_result.source_chunks,
        )
        
        # Stage 3: Async stream generate
        stream = self._chat_service.chat_stream_async(
            session=rag_session,
            provider=provider,
            fallback=True,
            model=model,
        )
        
        return retrieval_result, stream
    
    # ==================== Utility Methods ====================
    
    def get_active_providers(self) -> List[str]:
        """Get list of active LLM providers for RAG."""
        return [p.value for p in self._chat_service.get_active_providers()]
    
    def get_provider_count(self) -> int:
        """Get number of LLM providers configured for RAG."""
        return self._chat_service.get_provider_count()


# ==================== Global Singleton ====================

_rag_service: Optional[RAGService] = None
_rag_lock = threading.Lock()


def get_rag_service(
    config: Optional[RAGConfig] = None,
    retrieval_service: Optional[RetrievalService] = None,
    force_new: bool = False,
) -> RAGService:
    """
    Get the global RAGService singleton.
    
    Args:
        config: RAG configuration (uses defaults if omitted)
        retrieval_service: Pre-configured RetrievalService
        force_new: Force creation of a new instance
        
    Returns:
        RAGService singleton instance
    """
    global _rag_service
    
    with _rag_lock:
        if _rag_service is None or force_new:
            _rag_service = RAGService(
                config=config,
                retrieval_service=retrieval_service,
            )
        return _rag_service
