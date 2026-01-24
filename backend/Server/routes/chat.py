"""
Chat API routes.

Provides endpoints for chat inference with provider rotation and fallback.
"""

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator
import json
import logging

from ChatServer import (
    ChatSession,
    LLMProvider,
    chat_inference,
    chat_inference_stream,
    chat_inference_async,
    chat_inference_stream_async,
    get_chat_service,
    AllProvidersFailedError,
    AllKeysFailedError,
)
from ..models.schemas import (
    ChatRequest,
    ChatResponse,
    TokenUsage,
    ErrorResponse,
    MessageRole,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat"])


def _build_session(request: ChatRequest) -> ChatSession:
    """
    Build a ChatSession from the request.
    
    Args:
        request: The chat request containing messages and system prompt.
        
    Returns:
        A configured ChatSession instance.
    """
    session = ChatSession()
    
    # Add system prompt if provided
    if request.system_prompt:
        session.add_system_prompt(request.system_prompt)
    
    # Add messages
    for msg in request.messages:
        if msg.role == MessageRole.USER:
            session.add_user_message(msg.content)
        elif msg.role == MessageRole.ASSISTANT:
            session.add_assistant_message(msg.content)
        elif msg.role == MessageRole.SYSTEM:
            session.add_system_prompt(msg.content)
    
    return session


def _get_provider_enum(provider: str | None) -> LLMProvider | None:
    """Convert provider string to LLMProvider enum."""
    if provider is None:
        return None
    return LLMProvider(provider.upper())


@router.post(
    "",
    response_model=ChatResponse,
    responses={
        200: {"description": "Successful response"},
        503: {"model": ErrorResponse, "description": "All providers failed"},
    },
    summary="Chat with LLM",
    description="Send a chat request and get a response. Automatically rotates between providers.",
)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Send a chat request and get a response.
    
    The service automatically rotates between available providers (Groq, Cerebras)
    to distribute rate limits. If fallback is enabled, it will try all providers
    before failing.
    """
    session = _build_session(request)
    
    # Get provider enum if specified
    provider = None
    if request.provider:
        provider = LLMProvider(request.provider.value.upper())
    
    try:
        # Use async version for FastAPI
        response = await chat_inference_async(
            session,
            provider=provider,
            fallback=request.fallback,
            model=request.model,
        )
        
        # Get current provider info
        service = get_chat_service()
        current_provider = service.get_current_provider()
        
        # Use model from response or default model
        model_used = request.model if request.model else current_provider.default_model
        
        return ChatResponse(
            content=response.content,
            provider=response.provider or current_provider.provider.value.lower(),
            model=response.model,
            tokens=TokenUsage(
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                total_tokens=response.total_tokens,
            ),
        )
    except AllKeysFailedError as e:
        logger.error(f"All API keys failed for model {e.model}: {e.errors}")
        raise HTTPException(
            status_code=503,
            detail={
                "error": f"All API keys failed for {e.provider} with model {e.model}",
                "details": e.errors,
            },
        )
    except AllProvidersFailedError as e:
        logger.error(f"All providers failed: {e.errors}")
        raise HTTPException(
            status_code=503,
            detail={
                "error": "All providers failed",
                "details": e.errors,
            },
        )
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": str(e)},
        )


@router.post(
    "/stream",
    responses={
        200: {"description": "Streaming response"},
        503: {"model": ErrorResponse, "description": "All providers failed"},
    },
    summary="Stream chat response",
    description="Send a chat request and stream the response in real-time.",
)
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """
    Stream a chat response.
    
    Returns a streaming response with Server-Sent Events (SSE) format.
    Each chunk is sent as a JSON object with 'content' and 'done' fields.
    """
    session = _build_session(request)
    
    provider = None
    if request.provider:
        provider = LLMProvider(request.provider.value.upper())
    
    async def generate() -> AsyncGenerator[str, None]:
        try:
            async for chunk in chat_inference_stream_async(
                session,
                provider=provider,
                fallback=request.fallback,
                model=request.model,
            ):
                # Send as Server-Sent Events format
                data = json.dumps({"content": chunk, "done": False})
                yield f"data: {data}\n\n"
            
            # Send final message
            yield f"data: {json.dumps({'content': '', 'done': True})}\n\n"
            
        except AllKeysFailedError as e:
            error_data = json.dumps({
                "error": f"All API keys failed for {e.provider} with model {e.model}",
                "details": e.errors,
                "done": True,
            })
            yield f"data: {error_data}\n\n"
        except AllProvidersFailedError as e:
            error_data = json.dumps({
                "error": "All providers failed",
                "details": e.errors,
                "done": True,
            })
            yield f"data: {error_data}\n\n"
        except Exception as e:
            error_data = json.dumps({
                "error": str(e),
                "done": True,
            })
            yield f"data: {error_data}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.websocket("/ws")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocket endpoint for real-time chat.
    
    Accepts JSON messages with the same format as ChatRequest.
    Streams responses back as JSON chunks.
    """
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_json()
            
            try:
                # Parse request
                request = ChatRequest(**data)
                session = _build_session(request)
                
                provider = None
                if request.provider:
                    provider = LLMProvider(request.provider.value.upper())
                
                # Stream response
                async for chunk in chat_inference_stream_async(
                    session,
                    provider=provider,
                    fallback=request.fallback,
                    model=request.model,
                ):
                    await websocket.send_json({
                        "type": "chunk",
                        "content": chunk,
                    })
                
                # Send completion message
                await websocket.send_json({
                    "type": "done",
                    "content": "",
                })
                
            except AllKeysFailedError as e:
                await websocket.send_json({
                    "type": "error",
                    "error": f"All API keys failed for {e.provider} with model {e.model}",
                    "details": e.errors,
                })
            except AllProvidersFailedError as e:
                await websocket.send_json({
                    "type": "error",
                    "error": "All providers failed",
                    "details": e.errors,
                })
            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "error": str(e),
                })
                
    except WebSocketDisconnect:
        logger.info("WebSocket connection closed")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close(code=1011, reason=str(e))
