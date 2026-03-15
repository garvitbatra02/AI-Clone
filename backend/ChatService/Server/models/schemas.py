"""Pydantic models for API request/response schemas."""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class ProviderEnum(str, Enum):
    """Available LLM providers."""
    GROQ = "groq"
    CEREBRAS = "cerebras"
    COHERE = "cohere"


class MessageRole(str, Enum):
    """Message role types."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(BaseModel):
    """A single message in a conversation."""
    role: MessageRole
    content: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "role": "user",
                "content": "What is the capital of France?"
            }
        }


class ChatRequest(BaseModel):
    """Request body for chat endpoints."""
    messages: list[Message] = Field(
        ...,
        description="List of messages in the conversation",
        min_length=1
    )
    system_prompt: Optional[str] = Field(
        default=None,
        description="Optional system prompt to set assistant behavior"
    )
    provider: Optional[ProviderEnum] = Field(
        default=None,
        description="Specific provider to use (skips rotation if provided)"
    )
    model: Optional[str] = Field(
        default=None,
        description="Optional specific model name. If None, uses default models with provider rotation. "
                    "If specified, uses key-level fallback for that model."
    )
    fallback: bool = Field(
        default=True,
        description="If True, automatically try next providers on failure"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "messages": [
                    {"role": "user", "content": "What is the capital of France?"}
                ],
                "system_prompt": "You are a helpful assistant.",
                "fallback": True
            }
        }


class TokenUsage(BaseModel):
    """Token usage statistics."""
    prompt_tokens: int = Field(default=0, description="Tokens in the prompt")
    completion_tokens: int = Field(default=0, description="Tokens in the completion")
    total_tokens: int = Field(default=0, description="Total tokens used")


class ChatResponse(BaseModel):
    """Response body for chat endpoints."""
    content: str = Field(..., description="The assistant's response")
    provider: str = Field(..., description="The provider that served the request")
    model: str = Field(..., description="The model that generated the response")
    tokens: Optional[TokenUsage] = Field(default=None, description="Token usage statistics")
    
    class Config:
        json_schema_extra = {
            "example": {
                "content": "The capital of France is Paris.",
                "provider": "groq",
                "model": "llama-3.3-70b-versatile",
                "tokens": {
                    "prompt_tokens": 25,
                    "completion_tokens": 10,
                    "total_tokens": 35
                }
            }
        }


class StreamChunk(BaseModel):
    """A single chunk in a streaming response."""
    content: str = Field(..., description="The content chunk")
    done: bool = Field(default=False, description="Whether this is the final chunk")


class ErrorResponse(BaseModel):
    """Error response body."""
    error: str = Field(..., description="Error message")
    details: Optional[dict] = Field(default=None, description="Additional error details")
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "All providers failed",
                "details": {
                    "groq": "Rate limit exceeded",
                    "cerebras": "Connection timeout"
                }
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    providers: list[str] = Field(..., description="Available providers")
    provider_count: int = Field(..., description="Number of active providers")


class ServiceInfoResponse(BaseModel):
    """Service information response."""
    name: str = Field(..., description="Service name")
    version: str = Field(..., description="API version")
    providers: list[str] = Field(..., description="Available providers")
    current_provider: str = Field(..., description="Current provider in rotation")


# ==================== RAG Schemas ====================


class SourceChunk(BaseModel):
    """A source document chunk from retrieval."""
    id: str = Field(..., description="Unique identifier of the chunk")
    content: str = Field(..., description="Text content of the chunk")
    score: float = Field(..., description="Relevance score (higher = more relevant)")
    metadata: dict = Field(default_factory=dict, description="Chunk metadata (source, page, etc.)")


class RAGChatRequest(BaseModel):
    """Request body for RAG chat endpoints."""
    messages: list[Message] = Field(
        ...,
        description="List of messages in the conversation",
        min_length=1,
    )
    collection_name: Optional[str] = Field(
        default=None,
        description="Vector DB collection to search for context. "
                    "If omitted, uses the server's DEFAULT_COLLECTION.",
    )
    system_prompt: Optional[str] = Field(
        default=None,
        description="Additional system prompt (appended after RAG context)",
    )
    provider: Optional[ProviderEnum] = Field(
        default=None,
        description="Force a specific LLM provider (default: auto with priority Cohere → Cerebras → Groq)",
    )
    model: Optional[str] = Field(
        default=None,
        description="Force a specific model name",
    )
    top_k: Optional[int] = Field(
        default=None,
        description="Number of vector search candidates (default: 20)",
    )
    rerank: bool = Field(
        default=True,
        description="Whether to rerank results with Cohere Reranker",
    )
    rerank_top_n: Optional[int] = Field(
        default=None,
        description="Number of results to keep after reranking (default: 5)",
    )
    score_threshold: Optional[float] = Field(
        default=None,
        description="Minimum similarity score for vector search",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "messages": [
                    {"role": "user", "content": "What does the document say about machine learning?"}
                ],
                "collection_name": "my_docs",
                "rerank": True,
            }
        }


class RAGChatResponse(BaseModel):
    """Response body for RAG chat endpoints."""
    content: str = Field(..., description="The assistant's response")
    provider: str = Field(..., description="The LLM provider that served the request")
    model: str = Field(..., description="The model that generated the response")
    sources: list[SourceChunk] = Field(
        default_factory=list,
        description="Source chunks used to generate the answer",
    )
    reranked: bool = Field(
        default=False,
        description="Whether results were reranked",
    )
    tokens: Optional[TokenUsage] = Field(
        default=None,
        description="Token usage statistics",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "content": "According to the documents, machine learning is...",
                "provider": "cohere",
                "model": "command-a-03-2025",
                "sources": [
                    {
                        "id": "abc123",
                        "content": "Machine learning is a subset of AI...",
                        "score": 0.95,
                        "metadata": {"source": "ml_guide.pdf", "page": 1},
                    }
                ],
                "reranked": True,
            }
        }


class CollectionInfoItem(BaseModel):
    """Information about a single vector DB collection."""
    name: str = Field(..., description="Collection name")
    vectors_count: int = Field(..., description="Number of vectors in the collection")
    dimension: int = Field(default=0, description="Vector dimension")


class CollectionListResponse(BaseModel):
    """Response body for listing collections."""
    collections: list[CollectionInfoItem] = Field(
        default_factory=list,
        description="List of available collections with metadata",
    )
    total: int = Field(default=0, description="Total number of collections")


class RAGSearchRequest(BaseModel):
    """Request body for RAG search-only endpoint (no LLM generation)."""
    query: str = Field(..., description="Search query")
    collection_name: Optional[str] = Field(
        default=None,
        description="Vector DB collection to search. "
                    "If omitted, uses the server's DEFAULT_COLLECTION.",
    )
    top_k: Optional[int] = Field(
        default=None,
        description="Number of vector search candidates (default: 20)",
    )
    rerank: bool = Field(
        default=True,
        description="Whether to rerank results",
    )
    rerank_top_n: Optional[int] = Field(
        default=None,
        description="Number of results after reranking (default: 5)",
    )
    score_threshold: Optional[float] = Field(
        default=None,
        description="Minimum similarity score",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is machine learning?",
                "collection_name": "my_docs",
                "rerank": True,
            }
        }


class RAGSearchResponse(BaseModel):
    """Response body for RAG search-only endpoint."""
    query: str = Field(..., description="The original search query")
    results: list[SourceChunk] = Field(
        default_factory=list,
        description="Ranked search results",
    )
    reranked: bool = Field(
        default=False,
        description="Whether results were reranked",
    )
    total_candidates: int = Field(
        default=0,
        description="Total candidates from vector search before reranking",
    )
    context: str = Field(
        default="",
        description="Formatted context string (same as what the LLM would receive)",
    )
