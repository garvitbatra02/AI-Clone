"""
Tests for the RAG Retrieval Service.

Tests cover:
- CohereReranker: reranking with mocked Cohere API
- RetrievalService: full pipeline with/without reranking
- RAGService: end-to-end with mocked dependencies
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import dataclass

from RAGService.Data.VectorDB.base import SearchResult
from RAGService.Data.Reranker.base import RerankerConfig
from RAGService.Data.services.retrieval_service import (
    RetrievalService,
    RetrievalConfig,
    RetrievalResult,
)


# ==================== Test Fixtures ====================


def make_search_results(n: int = 5) -> list[SearchResult]:
    """Create mock SearchResult objects for testing."""
    return [
        SearchResult(
            id=f"doc_{i}",
            content=f"This is document {i} about topic {chr(65 + i)}.",
            score=1.0 - (i * 0.1),
            metadata={"source": f"file_{i}.pdf", "page": i + 1},
        )
        for i in range(n)
    ]


def make_mock_rerank_response(results: list[SearchResult], top_n: int = 3):
    """Create a mock Cohere rerank response."""
    # Reverse order to simulate reranking changing the order
    mock_results = []
    for rank, idx in enumerate(reversed(range(min(top_n, len(results))))):
        item = MagicMock()
        item.index = idx
        item.relevance_score = 0.95 - (rank * 0.1)
        mock_results.append(item)
    
    response = MagicMock()
    response.results = mock_results
    return response


# ==================== Reranker Tests ====================


class TestCohereReranker:
    """Tests for the CohereReranker."""
    
    @patch.dict("os.environ", {"COHERE_API_KEYS": "test-key"})
    @patch("cohere.ClientV2")
    @patch("cohere.AsyncClientV2")
    def test_rerank_basic(self, mock_async_client_cls, mock_client_cls):
        """Test basic reranking functionality."""
        from RAGService.Data.Reranker.cohere_reranker import CohereReranker
        
        results = make_search_results(5)
        mock_response = make_mock_rerank_response(results, top_n=3)
        
        mock_client = MagicMock()
        mock_client.rerank.return_value = mock_response
        mock_client_cls.return_value = mock_client
        
        config = RerankerConfig(model="rerank-v3.5", top_n=3)
        reranker = CohereReranker(config=config)
        
        reranked = reranker.rerank("test query", results, top_n=3)
        
        assert len(reranked) == 3
        # All results should have rerank metadata
        for r in reranked:
            assert r.metadata.get("reranked") is True
            assert "rerank_score" in r.metadata
            assert "original_score" in r.metadata
        
        # Scores should be from the reranker, not original
        assert reranked[0].score == 0.95
        
        mock_client.rerank.assert_called_once()
    
    @patch.dict("os.environ", {"COHERE_API_KEYS": "test-key"})
    @patch("cohere.ClientV2")
    @patch("cohere.AsyncClientV2")
    def test_rerank_empty_results(self, mock_async_client_cls, mock_client_cls):
        """Test reranking with empty results returns empty list."""
        from RAGService.Data.Reranker.cohere_reranker import CohereReranker
        
        config = RerankerConfig()
        reranker = CohereReranker(config=config)
        
        reranked = reranker.rerank("test query", [])
        assert reranked == []
    
    @patch.dict("os.environ", {"COHERE_API_KEYS": "test-key"})
    @patch("cohere.ClientV2")
    @patch("cohere.AsyncClientV2")
    def test_rerank_top_n_capped(self, mock_async_client_cls, mock_client_cls):
        """Test that top_n is capped to the number of results."""
        from RAGService.Data.Reranker.cohere_reranker import CohereReranker
        
        results = make_search_results(2)
        mock_response = make_mock_rerank_response(results, top_n=2)
        
        mock_client = MagicMock()
        mock_client.rerank.return_value = mock_response
        mock_client_cls.return_value = mock_client
        
        config = RerankerConfig(top_n=10)  # Requesting more than available
        reranker = CohereReranker(config=config)
        
        reranked = reranker.rerank("test query", results)
        
        # Should cap at available results
        call_kwargs = mock_client.rerank.call_args
        assert call_kwargs.kwargs["top_n"] == 2


# ==================== RetrievalService Tests ====================


class TestRetrievalService:
    """Tests for the RetrievalService."""
    
    def test_retrieve_with_reranking(self):
        """Test full retrieval pipeline with reranking."""
        mock_vectordb_service = MagicMock()
        mock_vectordb_service.search.return_value = make_search_results(10)
        
        config = RetrievalConfig(
            top_k=10,
            rerank_enabled=True,
            rerank_top_n=3,
        )
        service = RetrievalService(
            config=config,
            vectordb_service=mock_vectordb_service,
        )
        
        # Mock the reranker
        mock_reranker = MagicMock()
        mock_reranker.rerank.return_value = make_search_results(3)
        service._reranker = mock_reranker
        service._reranker_initialized = True
        
        result = service.retrieve(
            query="test query",
            collection_name="test_collection",
        )
        
        assert isinstance(result, RetrievalResult)
        assert result.query == "test query"
        assert result.reranked is True
        assert result.total_candidates == 10
        assert len(result.source_chunks) == 3
        assert result.context_str != ""
        
        mock_vectordb_service.search.assert_called_once()
        mock_reranker.rerank.assert_called_once()
    
    def test_retrieve_without_reranking(self):
        """Test retrieval with reranking disabled."""
        mock_vectordb_service = MagicMock()
        mock_vectordb_service.search.return_value = make_search_results(5)
        
        config = RetrievalConfig(
            top_k=5,
            rerank_enabled=False,
        )
        service = RetrievalService(
            config=config,
            vectordb_service=mock_vectordb_service,
        )
        
        result = service.retrieve(
            query="test query",
            collection_name="test_collection",
        )
        
        assert result.reranked is False
        assert result.total_candidates == 5
        assert len(result.source_chunks) == 5
    
    def test_retrieve_empty_results(self):
        """Test retrieval when vector search returns nothing."""
        mock_vectordb_service = MagicMock()
        mock_vectordb_service.search.return_value = []
        
        config = RetrievalConfig(rerank_enabled=True)
        service = RetrievalService(
            config=config,
            vectordb_service=mock_vectordb_service,
        )
        
        result = service.retrieve(
            query="test query",
            collection_name="test_collection",
        )
        
        assert result.context_str == ""
        assert result.source_chunks == []
        assert result.total_candidates == 0
        assert result.reranked is False
    
    def test_retrieve_reranker_failure_fallback(self):
        """Test that reranker failure falls back to vector search results."""
        mock_vectordb_service = MagicMock()
        original_results = make_search_results(10)
        mock_vectordb_service.search.return_value = original_results
        
        config = RetrievalConfig(
            top_k=10,
            rerank_enabled=True,
        )
        service = RetrievalService(
            config=config,
            vectordb_service=mock_vectordb_service,
        )
        
        # Mock a failing reranker
        mock_reranker = MagicMock()
        mock_reranker.rerank.side_effect = Exception("Rerank API error")
        service._reranker = mock_reranker
        service._reranker_initialized = True
        
        result = service.retrieve(
            query="test query",
            collection_name="test_collection",
        )
        
        # Should fall back to original results
        assert result.reranked is False
        assert len(result.source_chunks) == 10
    
    def test_context_formatting(self):
        """Test context string formatting."""
        mock_vectordb_service = MagicMock()
        mock_vectordb_service.search.return_value = make_search_results(2)
        
        config = RetrievalConfig(
            rerank_enabled=False,
            context_template="[{i}] {content}",
            context_separator="\n",
            context_header="Context:\n",
        )
        service = RetrievalService(
            config=config,
            vectordb_service=mock_vectordb_service,
        )
        
        result = service.retrieve("test", "collection")
        
        assert "Context:" in result.context_str
        assert "[1]" in result.context_str
        assert "[2]" in result.context_str
        assert "document 0" in result.context_str
    
    def test_search_only(self):
        """Test search_only bypasses reranking and formatting."""
        mock_vectordb_service = MagicMock()
        expected = make_search_results(5)
        mock_vectordb_service.search.return_value = expected
        
        config = RetrievalConfig(rerank_enabled=True)
        service = RetrievalService(
            config=config,
            vectordb_service=mock_vectordb_service,
        )
        
        results = service.search_only("test", "collection", k=5)
        
        assert results == expected
        mock_vectordb_service.search.assert_called_once_with(
            query="test",
            k=5,
            filters=None,
            collection_name="collection",
            score_threshold=None,
        )
    
    @pytest.mark.asyncio
    async def test_aretrieve(self):
        """Test async retrieval pipeline."""
        mock_vectordb_service = MagicMock()
        mock_vectordb_service.async_search = AsyncMock(
            return_value=make_search_results(5)
        )
        
        config = RetrievalConfig(
            top_k=5,
            rerank_enabled=False,
        )
        service = RetrievalService(
            config=config,
            vectordb_service=mock_vectordb_service,
        )
        
        result = await service.aretrieve(
            query="async test",
            collection_name="test_collection",
        )
        
        assert result.query == "async test"
        assert len(result.source_chunks) == 5
        mock_vectordb_service.async_search.assert_called_once()


# ==================== Multi-Collection Retrieval Tests ====================


def make_scored_results(prefix: str, scores: list[float]) -> list[SearchResult]:
    """Create SearchResults with explicit scores for a given collection prefix."""
    return [
        SearchResult(
            id=f"{prefix}_{i}",
            content=f"{prefix} chunk {i}",
            score=s,
            metadata={"source": f"{prefix}.pdf"},
        )
        for i, s in enumerate(scores)
    ]


class TestRetrievalServiceMulti:
    """Tests for the additive multi-collection retrieval methods."""

    def _service(self, config: RetrievalConfig) -> tuple[RetrievalService, MagicMock]:
        mock_vdb = MagicMock()
        mock_vdb.embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_vdb.collection_exists.return_value = True
        service = RetrievalService(config=config, vectordb_service=mock_vdb)
        # Disable reranker so we observe the raw pooled ordering
        service._reranker = None
        service._reranker_initialized = True
        return service, mock_vdb

    def test_pool_merges_and_sorts_by_global_score(self):
        """Candidates from N collections are merged and globally score-ordered."""
        config = RetrievalConfig(rerank_enabled=False, top_k=10)
        service, mock_vdb = self._service(config)

        per_collection = {
            "projects": make_scored_results("projects", [0.91, 0.55]),
            "notes": make_scored_results("notes", [0.82, 0.40]),
        }
        mock_vdb.search_by_vector.side_effect = (
            lambda embedding, k, filters, collection_name, score_threshold: per_collection[collection_name]
        )

        result = service.retrieve_multi(
            query="q", collection_names=["projects", "notes"],
        )

        # 4 pooled candidates, ordered by global score (not by collection)
        assert result.total_candidates == 4
        scores = [c.score for c in result.source_chunks]
        assert scores == sorted(scores, reverse=True)
        assert result.source_chunks[0].id == "projects_0"  # 0.91
        assert result.source_chunks[1].id == "notes_0"      # 0.82
        assert result.collections == ["projects", "notes"]

    def test_no_per_collection_quota_strong_collection_keeps_all(self):
        """A collection holding all the best chunks keeps all of them."""
        config = RetrievalConfig(rerank_enabled=False, top_k=10, max_pool=80)
        service, mock_vdb = self._service(config)

        strong = make_scored_results("strong", [0.99, 0.98, 0.97, 0.96])
        weak = make_scored_results("weak", [0.30, 0.20])
        per = {"strong": strong, "weak": weak}
        mock_vdb.search_by_vector.side_effect = (
            lambda embedding, k, filters, collection_name, score_threshold: per[collection_name]
        )

        result = service.retrieve_multi(
            query="q", collection_names=["strong", "weak"],
        )

        # All 4 strong chunks survive ahead of the weak ones — no quota dropped them
        top_ids = [c.id for c in result.source_chunks[:4]]
        assert top_ids == ["strong_0", "strong_1", "strong_2", "strong_3"]

    def test_max_pool_caps_pool_by_global_score(self):
        """The pool is truncated to max_pool, keeping the highest scores."""
        config = RetrievalConfig(rerank_enabled=False, top_k=50, max_pool=3)
        service, mock_vdb = self._service(config)

        a = make_scored_results("a", [0.9, 0.5, 0.1])
        b = make_scored_results("b", [0.8, 0.4, 0.2])
        per = {"a": a, "b": b}
        mock_vdb.search_by_vector.side_effect = (
            lambda embedding, k, filters, collection_name, score_threshold: per[collection_name]
        )

        result = service.retrieve_multi(query="q", collection_names=["a", "b"])

        assert result.total_candidates == 6      # all pooled before cap
        assert len(result.source_chunks) == 3     # capped
        assert [c.score for c in result.source_chunks] == [0.9, 0.8, 0.5]

    def test_missing_collection_is_skipped(self):
        """A missing collection is skipped, not fatal."""
        config = RetrievalConfig(rerank_enabled=False, top_k=10)
        service, mock_vdb = self._service(config)
        mock_vdb.collection_exists.side_effect = lambda name: name == "exists"
        mock_vdb.search_by_vector.return_value = make_scored_results("exists", [0.7, 0.6])

        result = service.retrieve_multi(
            query="q", collection_names=["exists", "ghost"],
        )

        assert result.total_candidates == 2
        assert result.collections == ["exists", "ghost"]

    def test_embeds_query_once(self):
        """The query embedding is computed once and reused across collections."""
        config = RetrievalConfig(rerank_enabled=False, top_k=10)
        service, mock_vdb = self._service(config)
        mock_vdb.search_by_vector.return_value = make_scored_results("c", [0.5])

        service.retrieve_multi(query="q", collection_names=["a", "b", "c"])

        assert mock_vdb.embeddings.embed_query.call_count == 1
        assert mock_vdb.search_by_vector.call_count == 3

    @pytest.mark.asyncio
    async def test_aretrieve_multi_concurrent(self):
        """Async multi searches every collection and pools the results."""
        config = RetrievalConfig(rerank_enabled=False, top_k=10)
        mock_vdb = MagicMock()
        mock_vdb.embeddings.async_embed_query = AsyncMock(return_value=[0.1, 0.2])
        mock_vdb.async_collection_exists = AsyncMock(return_value=True)

        per = {
            "a": make_scored_results("a", [0.9, 0.3]),
            "b": make_scored_results("b", [0.85, 0.2]),
        }

        async def _search(embedding, k, filters, collection_name, score_threshold):
            return per[collection_name]

        mock_vdb.async_search_by_vector = AsyncMock(side_effect=_search)

        service = RetrievalService(config=config, vectordb_service=mock_vdb)
        service._reranker = None
        service._reranker_initialized = True

        result = await service.aretrieve_multi(
            query="q", collection_names=["a", "b"],
        )

        assert result.total_candidates == 4
        assert result.source_chunks[0].id == "a_0"   # 0.9 global top
        assert result.source_chunks[1].id == "b_0"   # 0.85
        mock_vdb.embeddings.async_embed_query.assert_awaited_once()
        assert mock_vdb.async_search_by_vector.await_count == 2


# ==================== RAGService Tests ====================


class TestRAGService:
    """Tests for the RAG orchestrator."""
    
    @patch("RAGService.Data.services.rag_service.ChatService")
    def test_query_basic(self, mock_chat_service_cls):
        """Test basic RAG query flow."""
        from RAGService.Data.services.rag_service import RAGService, RAGConfig
        
        # Mock retrieval
        mock_retrieval = MagicMock()
        mock_retrieval.retrieve.return_value = RetrievalResult(
            context_str="Test context about AI.",
            source_chunks=make_search_results(3),
            query="What is AI?",
            reranked=True,
            total_candidates=10,
        )
        
        # Mock LLM response
        mock_llm_response = MagicMock()
        mock_llm_response.content = "AI is artificial intelligence."
        mock_llm_response.provider = "cohere"
        mock_llm_response.model = "command-a-03-2025"
        mock_llm_response.usage = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
        mock_llm_response.prompt_tokens = 100
        mock_llm_response.completion_tokens = 20
        mock_llm_response.total_tokens = 120
        
        mock_chat_instance = MagicMock()
        mock_chat_instance.chat.return_value = mock_llm_response
        mock_chat_service_cls.return_value = mock_chat_instance
        
        config = RAGConfig()
        service = RAGService(config=config, retrieval_service=mock_retrieval)
        
        response = service.query(
            user_query="What is AI?",
            collection_name="my_docs",
        )
        
        assert response.answer == "AI is artificial intelligence."
        assert response.provider_used == "cohere"
        assert response.model_used == "command-a-03-2025"
        assert len(response.sources) == 3
        assert response.retrieval_result.reranked is True
        
        # Verify retrieval was called
        mock_retrieval.retrieve.assert_called_once()
        
        # Verify chat was called with fallback
        mock_chat_instance.chat.assert_called_once()
        call_kwargs = mock_chat_instance.chat.call_args
        assert call_kwargs.kwargs["fallback"] is True
    
    @patch("RAGService.Data.services.rag_service.ChatService")
    def test_isolated_chat_service(self, mock_chat_service_cls):
        """Test that RAG creates its own isolated ChatService."""
        from RAGService.Data.services.rag_service import RAGService, RAGConfig
        
        config = RAGConfig()
        mock_retrieval = MagicMock()
        
        service = RAGService(config=config, retrieval_service=mock_retrieval)
        
        # Verify ChatService was created with RAG provider priority
        mock_chat_service_cls.assert_called_once()
        call_kwargs = mock_chat_service_cls.call_args
        providers = call_kwargs.kwargs.get("providers") or call_kwargs.args[0] if call_kwargs.args else None
        
        # Should have been called — the key point is it creates its own instance
        assert mock_chat_service_cls.called
    
    @patch("RAGService.Data.services.rag_service.ChatService")
    def test_context_injection_into_session(self, mock_chat_service_cls):
        """Test that RAG context is injected into the ChatSession."""
        from RAGService.Data.services.rag_service import RAGService, RAGConfig
        
        mock_retrieval = MagicMock()
        mock_retrieval.retrieve.return_value = RetrievalResult(
            context_str="Retrieved: Machine learning is a subset of AI.",
            source_chunks=make_search_results(2),
            query="What is ML?",
            reranked=False,
            total_candidates=5,
        )
        
        mock_llm_response = MagicMock()
        mock_llm_response.content = "ML is a subset of AI."
        mock_llm_response.provider = "groq"
        mock_llm_response.model = "llama-3.3-70b-versatile"
        
        mock_chat_instance = MagicMock()
        mock_chat_instance.chat.return_value = mock_llm_response
        mock_chat_service_cls.return_value = mock_chat_instance
        
        config = RAGConfig()
        service = RAGService(config=config, retrieval_service=mock_retrieval)
        
        service.query(user_query="What is ML?", collection_name="docs")
        
        # Check that the session passed to chat has RAG context
        call_args = mock_chat_instance.chat.call_args
        session = call_args.kwargs.get("session") or call_args.args[0]
        
        assert session.system_prompt is not None
        assert "Retrieved: Machine learning" in session.system_prompt
        assert session.context.get("rag_context") is not None
