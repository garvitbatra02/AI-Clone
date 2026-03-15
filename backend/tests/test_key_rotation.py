"""
Unit tests for the shared KeyRotationMixin.

Tests cover:
- KeyRotationMixin core logic (env loading, client caching, rotation loops)
- BaseEmbeddings key rotation (via CohereEmbeddings)
- BaseReranker key rotation (via CohereReranker)
- Backward compat (COHERE_API_KEY singular fallback)
- AllKeysFailedError on exhaustion
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock
from dataclasses import dataclass

from shared.key_rotation import KeyRotationMixin, AllKeysFailedError


# ============================================================================
# Fixtures
# ============================================================================


def make_search_results(n: int = 5):
    """Create mock SearchResult objects for testing."""
    from RAGService.Data.VectorDB.base import SearchResult
    return [
        SearchResult(
            id=f"doc_{i}",
            content=f"Document {i} content.",
            score=1.0 - (i * 0.1),
            metadata={"source": f"file_{i}.pdf"},
        )
        for i in range(n)
    ]


def make_mock_rerank_response(results, top_n: int = 3):
    """Create a mock Cohere rerank response."""
    mock_results = []
    for rank, idx in enumerate(reversed(range(min(top_n, len(results))))):
        item = MagicMock()
        item.index = idx
        item.relevance_score = 0.95 - (rank * 0.1)
        mock_results.append(item)
    response = MagicMock()
    response.results = mock_results
    return response


def make_mock_embed_response(n_embeddings: int = 1, dim: int = 4):
    """Create a mock Cohere embed response."""
    response = MagicMock()
    response.embeddings = MagicMock()
    response.embeddings.float_ = [[0.1 * j for j in range(dim)] for _ in range(n_embeddings)]
    return response


# ============================================================================
# AllKeysFailedError
# ============================================================================


class TestAllKeysFailedError:
    """Tests for the shared AllKeysFailedError exception."""

    def test_attributes(self):
        err = AllKeysFailedError("cohere", "embed-english-v3.0", {0: "rate limit", 1: "timeout"})
        assert err.service == "cohere"
        assert err.provider == "cohere"  # backward compat alias
        assert err.model == "embed-english-v3.0"
        assert err.errors == {0: "rate limit", 1: "timeout"}

    def test_message(self):
        err = AllKeysFailedError("groq", "llama", {0: "401"})
        assert "All API keys failed for groq" in str(err)
        assert "key_0: 401" in str(err)

    def test_importable_from_base_llm(self):
        """Backward compat: AllKeysFailedError still importable from ChatService.Chat.llm.base."""
        from ChatService.Chat.llm.base import AllKeysFailedError as AKF
        assert AKF is AllKeysFailedError

    def test_importable_from_embeddings_base(self):
        from RAGService.Data.Embeddings.base import AllKeysFailedError as AKF
        assert AKF is AllKeysFailedError

    def test_importable_from_reranker_base(self):
        from RAGService.Data.Reranker.base import AllKeysFailedError as AKF
        assert AKF is AllKeysFailedError


# ============================================================================
# KeyRotationMixin — standalone tests with a minimal subclass
# ============================================================================


class DummyService(KeyRotationMixin):
    """Minimal concrete class for testing the mixin directly."""
    ENV_VAR_NAME = "DUMMY_API_KEYS"

    def __init__(self, api_keys=None, max_retries=2):
        self._init_key_rotation(api_keys, max_retries)

    def _initialize_client(self, api_key: str) -> None:
        self._client = MagicMock(name=f"client_{api_key}")
        self._async_client = MagicMock(name=f"async_client_{api_key}")


class TestKeyRotationMixin:
    """Tests for the generic KeyRotationMixin logic."""

    def test_init_with_explicit_keys(self):
        svc = DummyService(api_keys=["k1", "k2"])
        assert svc._api_keys == ["k1", "k2"]
        assert 0 in svc._client_cache  # first key cached at init

    @patch.dict("os.environ", {"DUMMY_API_KEYS": "envkey1,envkey2,envkey3"})
    def test_init_from_env(self):
        svc = DummyService()
        assert svc._api_keys == ["envkey1", "envkey2", "envkey3"]

    def test_init_no_keys_raises(self):
        with pytest.raises(ValueError, match="No API keys available"):
            DummyService(api_keys=[])

    @patch.dict("os.environ", {}, clear=False)
    def test_init_no_env_raises(self):
        # Ensure DUMMY_API_KEYS is not set
        import os
        os.environ.pop("DUMMY_API_KEYS", None)
        with pytest.raises(ValueError, match="No API keys available"):
            DummyService()

    def test_client_cache_per_key(self):
        svc = DummyService(api_keys=["k1", "k2"])
        # Key 0 is cached at init
        svc._get_client_for_key(1)
        assert 1 in svc._client_cache
        # Both keys have separate clients
        assert svc._client_cache[0]["_client"] is not svc._client_cache[1]["_client"]

    def test_client_cache_restores(self):
        svc = DummyService(api_keys=["k1", "k2"])
        client_0 = svc._client
        async_client_0 = svc._async_client

        # Switch to key 1
        svc._get_client_for_key(1)
        client_1 = svc._client
        assert client_1 is not client_0

        # Switch back to key 0 — should restore
        svc._get_client_for_key(0)
        assert svc._client is client_0
        assert svc._async_client is async_client_0

    def test_execute_with_rotation_success_first_key(self):
        svc = DummyService(api_keys=["k1"])
        result = svc._execute_with_rotation(
            operation=lambda: "hello",
            is_valid=lambda r: bool(r),
            service_label="test",
            model_label="model",
        )
        assert result == "hello"

    def test_execute_with_rotation_fails_over_to_key2(self):
        svc = DummyService(api_keys=["k1", "k2"])
        call_count = {"n": 0}

        def operation():
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("key1 rate limited")
            return "from_key2"

        result = svc._execute_with_rotation(
            operation=operation,
            is_valid=lambda r: bool(r),
            service_label="test",
            model_label="model",
        )
        assert result == "from_key2"
        assert call_count["n"] == 2

    def test_execute_with_rotation_retries_on_invalid(self):
        svc = DummyService(api_keys=["k1"], max_retries=1)
        call_count = {"n": 0}

        def operation():
            call_count["n"] += 1
            if call_count["n"] <= 1:
                return None  # invalid
            return "valid"

        result = svc._execute_with_rotation(
            operation=operation,
            is_valid=lambda r: r is not None,
            service_label="test",
            model_label="model",
        )
        assert result == "valid"
        assert call_count["n"] == 2  # first invalid, retry succeeds

    def test_execute_with_rotation_all_keys_fail(self):
        svc = DummyService(api_keys=["k1", "k2"], max_retries=0)

        with pytest.raises(AllKeysFailedError) as exc_info:
            svc._execute_with_rotation(
                operation=lambda: (_ for _ in ()).throw(RuntimeError("fail")),
                is_valid=lambda r: True,
                service_label="test-svc",
                model_label="model-x",
            )
        assert exc_info.value.service == "test-svc"
        assert 0 in exc_info.value.errors
        assert 1 in exc_info.value.errors

    @pytest.mark.asyncio
    async def test_execute_with_rotation_async_success(self):
        svc = DummyService(api_keys=["k1"])

        async def op():
            return "async_hello"

        result = await svc._execute_with_rotation_async(
            operation=op,
            is_valid=lambda r: bool(r),
            service_label="test",
            model_label="model",
        )
        assert result == "async_hello"

    @pytest.mark.asyncio
    async def test_execute_with_rotation_async_failover(self):
        svc = DummyService(api_keys=["k1", "k2"])
        call_count = {"n": 0}

        async def op():
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("async key1 fail")
            return "async_key2"

        result = await svc._execute_with_rotation_async(
            operation=op,
            is_valid=lambda r: bool(r),
            service_label="test",
            model_label="model",
        )
        assert result == "async_key2"

    def test_stream_with_rotation_success(self):
        svc = DummyService(api_keys=["k1"])

        def op():
            yield "chunk1"
            yield "chunk2"

        chunks = list(svc._stream_with_rotation(
            operation=op,
            service_label="test",
            model_label="model",
        ))
        assert chunks == ["chunk1", "chunk2"]

    def test_stream_with_rotation_failover(self):
        svc = DummyService(api_keys=["k1", "k2"])
        call_count = {"n": 0}

        def op():
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("stream fail")
            yield "ok_chunk"

        chunks = list(svc._stream_with_rotation(
            operation=op,
            service_label="test",
            model_label="model",
        ))
        assert chunks == ["ok_chunk"]


# ============================================================================
# CohereEmbeddings — key rotation integration
# ============================================================================


class TestCohereEmbeddingsKeyRotation:
    """Test that CohereEmbeddings inherits key rotation from the mixin."""

    @patch.dict("os.environ", {"COHERE_API_KEYS": "key1,key2"})
    @patch("cohere.ClientV2")
    @patch("cohere.AsyncClientV2")
    def test_multiple_keys_loaded(self, mock_async_cls, mock_cls):
        from RAGService.Data.Embeddings.providers.cohere_embeddings import CohereEmbeddings
        from RAGService.Data.Embeddings.base import EmbeddingConfig, EmbeddingProvider

        config = EmbeddingConfig.for_cohere()
        emb = CohereEmbeddings(config)
        assert emb._api_keys == ["key1", "key2"]

    @patch.dict("os.environ", {"COHERE_API_KEY": "single-key"}, clear=False)
    @patch("cohere.ClientV2")
    @patch("cohere.AsyncClientV2")
    def test_backward_compat_singular_env(self, mock_async_cls, mock_cls):
        """COHERE_API_KEY (singular) should work as fallback."""
        import os
        os.environ.pop("COHERE_API_KEYS", None)

        from RAGService.Data.Embeddings.providers.cohere_embeddings import CohereEmbeddings
        from RAGService.Data.Embeddings.base import EmbeddingConfig

        config = EmbeddingConfig.for_cohere()
        emb = CohereEmbeddings(config)
        assert emb._api_keys == ["single-key"]

    @patch.dict("os.environ", {"COHERE_API_KEYS": "key1,key2"})
    @patch("cohere.ClientV2")
    @patch("cohere.AsyncClientV2")
    def test_embed_query_failover(self, mock_async_cls, mock_cls):
        """If key 0 fails, embed_query should succeed with key 1."""
        from RAGService.Data.Embeddings.providers.cohere_embeddings import CohereEmbeddings
        from RAGService.Data.Embeddings.base import EmbeddingConfig

        # Client for key 0 raises; client for key 1 succeeds
        client0 = MagicMock()
        client0.embed.side_effect = RuntimeError("rate limited")
        client1 = MagicMock()
        client1.embed.return_value = make_mock_embed_response(1, 4)
        mock_cls.side_effect = [client0, client1]

        config = EmbeddingConfig.for_cohere()
        emb = CohereEmbeddings(config)

        result = emb.embed_query("test")
        assert len(result) == 4
        # key 0 client was called and failed
        client0.embed.assert_called_once()
        # key 1 client succeeded
        client1.embed.assert_called_once()

    @patch.dict("os.environ", {"COHERE_API_KEYS": "key1,key2"})
    @patch("cohere.ClientV2")
    @patch("cohere.AsyncClientV2")
    def test_embed_documents_failover(self, mock_async_cls, mock_cls):
        """If key 0 fails, embed_documents should succeed with key 1."""
        from RAGService.Data.Embeddings.providers.cohere_embeddings import CohereEmbeddings
        from RAGService.Data.Embeddings.base import EmbeddingConfig

        client0 = MagicMock()
        client0.embed.side_effect = RuntimeError("rate limited")
        client1 = MagicMock()
        client1.embed.return_value = make_mock_embed_response(2, 4)
        mock_cls.side_effect = [client0, client1]

        config = EmbeddingConfig.for_cohere()
        emb = CohereEmbeddings(config)

        result = emb.embed_documents(["text1", "text2"])
        assert len(result) == 2
        assert len(result[0]) == 4

    @patch.dict("os.environ", {"COHERE_API_KEYS": "key1"})
    @patch("cohere.ClientV2")
    @patch("cohere.AsyncClientV2")
    def test_embed_query_all_keys_fail(self, mock_async_cls, mock_cls):
        """When all keys fail, AllKeysFailedError should be raised."""
        from RAGService.Data.Embeddings.providers.cohere_embeddings import CohereEmbeddings
        from RAGService.Data.Embeddings.base import EmbeddingConfig

        client = MagicMock()
        client.embed.side_effect = RuntimeError("permanent failure")
        mock_cls.return_value = client

        config = EmbeddingConfig.for_cohere(max_retries=0)
        emb = CohereEmbeddings(config)

        with pytest.raises(AllKeysFailedError) as exc_info:
            emb.embed_query("test")
        assert "permanent failure" in str(exc_info.value)

    @patch.dict("os.environ", {"COHERE_API_KEYS": "key1"})
    @patch("cohere.ClientV2")
    @patch("cohere.AsyncClientV2")
    def test_embed_documents_empty_returns_empty(self, mock_async_cls, mock_cls):
        """Empty texts list should return [] without hitting the API."""
        from RAGService.Data.Embeddings.providers.cohere_embeddings import CohereEmbeddings
        from RAGService.Data.Embeddings.base import EmbeddingConfig

        config = EmbeddingConfig.for_cohere()
        emb = CohereEmbeddings(config)
        assert emb.embed_documents([]) == []


# ============================================================================
# CohereReranker — key rotation integration
# ============================================================================


class TestCohereRerankerKeyRotation:
    """Test that CohereReranker inherits key rotation from the mixin."""

    @patch.dict("os.environ", {"COHERE_API_KEYS": "key1,key2,key3"})
    @patch("cohere.ClientV2")
    @patch("cohere.AsyncClientV2")
    def test_multiple_keys_loaded(self, mock_async_cls, mock_cls):
        from RAGService.Data.Reranker.cohere_reranker import CohereReranker
        from RAGService.Data.Reranker.base import RerankerConfig

        reranker = CohereReranker(RerankerConfig())
        assert reranker._api_keys == ["key1", "key2", "key3"]

    @patch.dict("os.environ", {"COHERE_API_KEY": "single-key"}, clear=False)
    @patch("cohere.ClientV2")
    @patch("cohere.AsyncClientV2")
    def test_backward_compat_singular_env(self, mock_async_cls, mock_cls):
        """COHERE_API_KEY (singular) should work as fallback."""
        import os
        os.environ.pop("COHERE_API_KEYS", None)

        from RAGService.Data.Reranker.cohere_reranker import CohereReranker
        from RAGService.Data.Reranker.base import RerankerConfig

        reranker = CohereReranker(RerankerConfig())
        assert reranker._api_keys == ["single-key"]

    @patch.dict("os.environ", {"COHERE_API_KEYS": "key1,key2"})
    @patch("cohere.ClientV2")
    @patch("cohere.AsyncClientV2")
    def test_rerank_failover(self, mock_async_cls, mock_cls):
        """If key 0 fails, rerank should succeed with key 1."""
        from RAGService.Data.Reranker.cohere_reranker import CohereReranker
        from RAGService.Data.Reranker.base import RerankerConfig

        results = make_search_results(5)
        mock_response = make_mock_rerank_response(results, top_n=3)

        client0 = MagicMock()
        client0.rerank.side_effect = RuntimeError("rate limited")
        client1 = MagicMock()
        client1.rerank.return_value = mock_response
        mock_cls.side_effect = [client0, client1]

        config = RerankerConfig(top_n=3)
        reranker = CohereReranker(config)

        reranked = reranker.rerank("test query", results, top_n=3)
        assert len(reranked) == 3
        # key 0 was tried and failed
        client0.rerank.assert_called_once()
        # key 1 succeeded
        client1.rerank.assert_called_once()

    @patch.dict("os.environ", {"COHERE_API_KEYS": "key1"})
    @patch("cohere.ClientV2")
    @patch("cohere.AsyncClientV2")
    def test_rerank_all_keys_fail(self, mock_async_cls, mock_cls):
        """When all keys fail, AllKeysFailedError should be raised."""
        from RAGService.Data.Reranker.cohere_reranker import CohereReranker
        from RAGService.Data.Reranker.base import RerankerConfig

        client = MagicMock()
        client.rerank.side_effect = RuntimeError("permanent failure")
        mock_cls.return_value = client

        config = RerankerConfig(max_retries=0)
        reranker = CohereReranker(config)

        results = make_search_results(3)
        with pytest.raises(AllKeysFailedError) as exc_info:
            reranker.rerank("test", results)
        assert "permanent failure" in str(exc_info.value)

    @patch.dict("os.environ", {"COHERE_API_KEYS": "key1"})
    @patch("cohere.ClientV2")
    @patch("cohere.AsyncClientV2")
    def test_rerank_empty_returns_empty(self, mock_async_cls, mock_cls):
        """Empty results should return [] without hitting the API."""
        from RAGService.Data.Reranker.cohere_reranker import CohereReranker

        reranker = CohereReranker()
        assert reranker.rerank("query", []) == []

    @patch.dict("os.environ", {"COHERE_API_KEYS": "key1"})
    @patch("cohere.ClientV2")
    @patch("cohere.AsyncClientV2")
    def test_reranker_config_api_key_compat(self, mock_async_cls, mock_cls):
        """Passing api_key (singular) in config should still work."""
        import os
        os.environ.pop("COHERE_API_KEYS", None)
        os.environ.pop("COHERE_API_KEY", None)

        from RAGService.Data.Reranker.cohere_reranker import CohereReranker
        from RAGService.Data.Reranker.base import RerankerConfig

        config = RerankerConfig(api_key="explicit-key")
        reranker = CohereReranker(config)
        assert reranker._api_keys == ["explicit-key"]

    @patch.dict("os.environ", {"COHERE_API_KEYS": "key1"})
    @patch("cohere.ClientV2")
    @patch("cohere.AsyncClientV2")
    def test_reranker_config_api_keys_compat(self, mock_async_cls, mock_cls):
        """Passing api_keys (plural) in config should work."""
        from RAGService.Data.Reranker.cohere_reranker import CohereReranker
        from RAGService.Data.Reranker.base import RerankerConfig

        config = RerankerConfig(api_keys=["ex1", "ex2"])
        reranker = CohereReranker(config)
        assert reranker._api_keys == ["ex1", "ex2"]


# ============================================================================
# Async reranker key rotation
# ============================================================================


class TestCohereRerankerAsyncKeyRotation:
    """Async key rotation for CohereReranker."""

    @pytest.mark.asyncio
    @patch.dict("os.environ", {"COHERE_API_KEYS": "key1,key2"})
    @patch("cohere.ClientV2")
    @patch("cohere.AsyncClientV2")
    async def test_arerank_failover(self, mock_async_cls, mock_cls):
        """If async key 0 fails, arerank should succeed with key 1."""
        from RAGService.Data.Reranker.cohere_reranker import CohereReranker
        from RAGService.Data.Reranker.base import RerankerConfig

        results = make_search_results(5)
        mock_response = make_mock_rerank_response(results, top_n=3)

        async_client0 = MagicMock()
        async_client0.rerank = AsyncMock(side_effect=RuntimeError("rate limited"))
        async_client1 = MagicMock()
        async_client1.rerank = AsyncMock(return_value=mock_response)
        mock_async_cls.side_effect = [async_client0, async_client1]

        config = RerankerConfig(top_n=3)
        reranker = CohereReranker(config)

        reranked = await reranker.arerank("test query", results, top_n=3)
        assert len(reranked) == 3
