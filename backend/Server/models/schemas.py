"""Pydantic models for API request/response schemas."""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class ProviderEnum(str, Enum):
    """Available LLM providers."""
    GROQ = "groq"
    CEREBRAS = "cerebras"


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
