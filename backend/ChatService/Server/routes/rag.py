"""
RAG API routes.

Provides endpoints for Retrieval Augmented Generation:
- /api/rag/chat: RAG chat with context retrieval + LLM generation
- /api/rag/chat/stream: Streaming RAG chat via SSE
- /api/rag/search: Retrieval-only (no LLM), returns ranked chunks
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator
import json
import logging
import os

from ChatService.Chat.llm.base import LLMProvider
from shared.key_rotation import AllKeysFailedError
from ChatService.Chat.services.chat_service import AllProvidersFailedError
from ..models.schemas import (
    RAGChatRequest,
    RAGChatResponse,
    RAGSearchRequest,
    RAGSearchResponse,
    CollectionInfoItem,
    CollectionListResponse,
    SourceChunk,
    TokenUsage,
    ErrorResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rag", tags=["RAG"])


def _get_rag_service():
    """Get the RAG service singleton."""
    from RAGService.Data.services.rag_service import get_rag_service
    return get_rag_service()


def _get_retrieval_service():
    """Get the retrieval service singleton."""
    from RAGService.Data.services.retrieval_service import get_retrieval_service
    return get_retrieval_service()


def _get_vectordb_service():
    """Get the VectorDB service from the retrieval service."""
    retrieval = _get_retrieval_service()
    return retrieval.vectordb_service


def _search_results_to_source_chunks(results) -> list[SourceChunk]:
    """Convert SearchResult objects to SourceChunk schema objects."""
    return [
        SourceChunk(
            id=r.id,
            content=r.content,
            score=r.score,
            metadata=r.metadata,
        )
        for r in results
    ]


def _resolve_collection(name: str | None) -> str:
    """Resolve collection name: explicit value → DEFAULT_COLLECTION env → 'assets'."""
    if name:
        return name
    default = os.environ.get("DEFAULT_COLLECTION", "").strip()
    if not default:
        raise HTTPException(
            status_code=400,
            detail={"error": "collection_name is required (no DEFAULT_COLLECTION configured)"},
        )
    return default


@router.post(
    "/chat",
    response_model=RAGChatResponse,
    responses={
        200: {"description": "Successful RAG response"},
        503: {"model": ErrorResponse, "description": "All providers failed"},
    },
    summary="RAG Chat",
    description=(
        "Send a query with retrieval-augmented generation. "
        "Searches the vector DB for relevant context, optionally reranks, "
        "then generates a response using the LLM with context injection. "
        "Provider priority: Cohere → Cerebras → Groq with auto-fallback."
    ),
)
async def rag_chat(request: RAGChatRequest) -> RAGChatResponse:
    """RAG chat endpoint: retrieve context + generate response."""
    rag_service = _get_rag_service()
    
    # Extract the user query (last user message)
    user_query = None
    for msg in reversed(request.messages):
        if msg.role.value == "user":
            user_query = msg.content
            break
    
    if not user_query:
        raise HTTPException(
            status_code=400,
            detail={"error": "At least one user message is required"},
        )
    
    # Determine provider
    provider = None
    if request.provider:
        provider = LLMProvider(request.provider.value)
    
    try:
        collection = _resolve_collection(request.collection_name)
        response = await rag_service.aquery(
            user_query=user_query,
            collection_name=collection,
            system_prompt=request.system_prompt,
            provider=provider,
            model=request.model,
            top_k=request.top_k,
            rerank_top_n=request.rerank_top_n if request.rerank else None,
        )
        
        # Build token usage
        tokens = None
        if response.llm_response and response.llm_response.usage:
            tokens = TokenUsage(
                prompt_tokens=response.llm_response.prompt_tokens,
                completion_tokens=response.llm_response.completion_tokens,
                total_tokens=response.llm_response.total_tokens,
            )
        
        return RAGChatResponse(
            content=response.answer,
            provider=response.provider_used,
            model=response.model_used,
            sources=_search_results_to_source_chunks(response.sources),
            reranked=response.retrieval_result.reranked,
            tokens=tokens,
        )
    
    except AllKeysFailedError as e:
        logger.error(f"All API keys failed for RAG: {e}")
        raise HTTPException(
            status_code=503,
            detail={
                "error": f"All API keys failed for {e.provider} with model {e.model}",
                "details": e.errors,
            },
        )
    except AllProvidersFailedError as e:
        logger.error(f"All RAG providers failed: {e.errors}")
        raise HTTPException(
            status_code=503,
            detail={"error": "All RAG providers failed", "details": e.errors},
        )
    except Exception as e:
        logger.error(f"RAG chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail={"error": str(e)})


@router.post(
    "/chat/stream",
    responses={
        200: {"description": "Streaming RAG response"},
        503: {"model": ErrorResponse, "description": "All providers failed"},
    },
    summary="Stream RAG Chat",
    description=(
        "Stream a RAG response. First retrieves context, then streams "
        "the LLM response via Server-Sent Events. Sources are sent in "
        "the first event before streaming begins."
    ),
)
async def rag_chat_stream(request: RAGChatRequest) -> StreamingResponse:
    """Streaming RAG chat endpoint."""
    rag_service = _get_rag_service()
    
    # Extract user query
    user_query = None
    for msg in reversed(request.messages):
        if msg.role.value == "user":
            user_query = msg.content
            break
    
    if not user_query:
        raise HTTPException(
            status_code=400,
            detail={"error": "At least one user message is required"},
        )
    
    provider = None
    if request.provider:
        provider = LLMProvider(request.provider.value)
    
    async def generate() -> AsyncGenerator[str, None]:
        try:
            collection = _resolve_collection(request.collection_name)
            retrieval_result, stream = await rag_service.aquery_stream(
                user_query=user_query,
                collection_name=collection,
                system_prompt=request.system_prompt,
                provider=provider,
                model=request.model,
                top_k=request.top_k,
                rerank_top_n=request.rerank_top_n if request.rerank else None,
            )
            
            # Send sources first
            sources_data = json.dumps({
                "type": "sources",
                "sources": [
                    {
                        "id": s.id,
                        "content": s.content,
                        "score": s.score,
                        "metadata": s.metadata,
                    }
                    for s in retrieval_result.source_chunks
                ],
                "reranked": retrieval_result.reranked,
                "done": False,
            })
            yield f"data: {sources_data}\n\n"
            
            # Stream LLM response
            async for chunk in stream:
                data = json.dumps({
                    "type": "content",
                    "content": chunk,
                    "done": False,
                })
                yield f"data: {data}\n\n"
            
            # Send done signal
            yield f"data: {json.dumps({'type': 'done', 'content': '', 'done': True})}\n\n"
        
        except AllKeysFailedError as e:
            error_data = json.dumps({
                "type": "error",
                "error": f"All API keys failed for {e.provider}",
                "details": e.errors,
                "done": True,
            })
            yield f"data: {error_data}\n\n"
        except AllProvidersFailedError as e:
            error_data = json.dumps({
                "type": "error",
                "error": "All RAG providers failed",
                "details": e.errors,
                "done": True,
            })
            yield f"data: {error_data}\n\n"
        except Exception as e:
            error_data = json.dumps({
                "type": "error",
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


@router.post(
    "/search",
    response_model=RAGSearchResponse,
    summary="RAG Search",
    description=(
        "Retrieval-only endpoint: searches the vector DB and optionally reranks, "
        "but does NOT generate an LLM response. Useful for debugging retrieval "
        "quality or building custom pipelines."
    ),
)
async def rag_search(request: RAGSearchRequest) -> RAGSearchResponse:
    """Retrieval-only endpoint (no LLM generation)."""
    retrieval_service = _get_retrieval_service()
    
    try:
        collection = _resolve_collection(request.collection_name)
        result = await retrieval_service.aretrieve(
            query=request.query,
            collection_name=collection,
            top_k=request.top_k,
            rerank_top_n=request.rerank_top_n if request.rerank else None,
            score_threshold=request.score_threshold,
        )
        
        return RAGSearchResponse(
            query=result.query,
            results=_search_results_to_source_chunks(result.source_chunks),
            reranked=result.reranked,
            total_candidates=result.total_candidates,
            context=result.context_str,
        )
    
    except Exception as e:
        logger.error(f"RAG search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail={"error": str(e)})


@router.get(
    "/collections",
    response_model=CollectionListResponse,
    summary="List Collections",
    description=(
        "List all available vector DB collections with their vector counts "
        "and dimensions. Useful for verifying uploaded documents are visible "
        "to the retrieval pipeline."
    ),
)
async def list_collections() -> CollectionListResponse:
    """List available vector DB collections with metadata."""
    try:
        vectordb_svc = _get_vectordb_service()
        collection_names = vectordb_svc.list_collections()
        
        items = []
        for name in collection_names:
            try:
                info = vectordb_svc.get_collection_info(collection_name=name)
                items.append(CollectionInfoItem(
                    name=name,
                    vectors_count=info.vector_count,
                    dimension=info.dimension,
                ))
            except Exception as e:
                logger.warning(f"Could not get info for collection '{name}': {e}")
                items.append(CollectionInfoItem(
                    name=name,
                    vectors_count=0,
                    dimension=0,
                ))
        
        return CollectionListResponse(
            collections=items,
            total=len(items),
        )
    except Exception as e:
        logger.error(f"List collections error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail={"error": str(e)})
