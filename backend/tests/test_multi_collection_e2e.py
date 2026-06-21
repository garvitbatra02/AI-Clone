"""
End-to-end tests for MULTI-COLLECTION RAG search.

Validates the additive multi-collection capability end-to-end:

    Upload distinct content into SEPARATE collections
      → search a single collection (isolation: can't see other collections)
      → search multiple collections (fix: pooled + globally re-ranked)
      → full RAG chat spanning collections

Exercises BOTH layers:
    • Service layer  — RetrievalService.retrieve_multi / aretrieve_multi,
                       RAGService routing (collection_names)
    • HTTP layer     — POST /api/rag/search and POST /api/rag/chat with
                       collection_names, via FastAPI TestClient

Design of the test corpus (3 separate collections, distinct topics):
    • collection "resume_*"  ← my all details.txt   (Microsoft, CGPA 9.55, FindOne)
    • collection "convos_*"  ← conversation.json     (Docker, REST vs GraphQL)
    • collection "training_*"← Personal_TrainingData.csv (education, contact info)

The three collections share ONE in-memory Qdrant client + ONE Cohere
embeddings instance, so scores are directly comparable across collections —
exactly the production assumption that makes global pooling valid.

Requires (.env):
    COHERE_API_KEY  — Cohere embed-english-v3.0, rerank-v3.5, command-a-03-2025

Run:
    PYTHONPATH=. python tests/test_multi_collection_e2e.py
    PYTHONPATH=. python -m pytest tests/test_multi_collection_e2e.py -v --tb=short
"""

from __future__ import annotations

import asyncio
import os
import sys
import traceback
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.testclient import TestClient

from RAGService.Data.VectorDB import (
    VectorDBFactory,
    VectorDBProvider,
    DistanceMetric,
)
from RAGService.Data.Embeddings import EmbeddingsFactory
from RAGService.Data.services.asset_upload_service import (
    AssetUploadConfig,
    AssetUploadService,
)
from RAGService.Data.services.vectordb_service import (
    VectorDBService,
    VectorDBServiceConfig,
)
from RAGService.Data.services.retrieval_service import (
    RetrievalService,
    RetrievalConfig,
)
from RAGService.Data.services.rag_service import RAGService

# ── Configuration ────────────────────────────────────────────────

EMBEDDING_DIM = 1024  # embed-english-v3.0
RUN_ID = uuid.uuid4().hex[:8]

DATA_DIR = Path(__file__).parent / "document_loaders_tests"
TXT_FILE = DATA_DIR / "my all details.txt"
CSV_FILE = DATA_DIR / "Personal_TrainingData.csv"
JSON_FILE = DATA_DIR / "conversation.json"

COLL_RESUME = f"resume_{RUN_ID}"
COLL_CONVOS = f"convos_{RUN_ID}"
COLL_TRAINING = f"training_{RUN_ID}"


# ── Pipeline (built once, shared across tests) ───────────────────

_pipeline: dict = {}


def _build_pipeline():
    """
    Build a 3-collection in-memory RAG pipeline backed by real Cohere.

    Returns (rag_service, retrieval_service, vectordb_service).
    Cached so the (slow) embedding/upload happens only once.
    """
    if _pipeline:
        return _pipeline["rag"], _pipeline["retrieval"], _pipeline["vectordb"]

    embeddings = EmbeddingsFactory.create_cohere(model_name="embed-english-v3.0")

    # One shared in-memory Qdrant client holds all three collections
    vectordb = VectorDBFactory.create_from_env(
        provider=VectorDBProvider.QDRANT,
        collection_name=COLL_RESUME,
        embedding_dimension=EMBEDDING_DIM,
        in_memory=True,
        distance_metric=DistanceMetric.COSINE,
    )

    vectordb_service = VectorDBService(
        config=VectorDBServiceConfig(
            collection_name=COLL_RESUME,
            in_memory=True,
            auto_create_collection=True,
        ),
        vectordb=vectordb,
        embeddings=embeddings,
    )

    # One uploader, shared client → distinct files into distinct collections
    upload_svc = AssetUploadService(
        config=AssetUploadConfig(
            default_collection=COLL_RESUME,
            chunk_size=1000,
            chunk_overlap=200,
            use_smart_chunker=True,
            use_llm_analysis=False,
        ),
        vectordb=vectordb,
        embeddings=embeddings,
    )

    uploads = [
        (COLL_RESUME, TXT_FILE),
        (COLL_CONVOS, JSON_FILE),
        (COLL_TRAINING, CSV_FILE),
    ]
    # Seed through the ASYNC upload path so the data lands in the same
    # async-client store the HTTP endpoints (and aretrieve_multi) query.
    # In-memory Qdrant keeps separate sync/async stores; production (cloud)
    # shares one server, so this only matters for the test harness.
    for collection, fpath in uploads:
        assert fpath.exists(), f"Missing test data file: {fpath}"
        result = asyncio.run(
            upload_svc.async_upload_file(fpath, collection_name=collection)
        )
        assert result.success, f"Upload failed for {fpath}: {result.error}"
        print(f"    ✅ {collection}: {result.total_chunks} chunks "
              f"({fpath.name})")

    retrieval_service = RetrievalService(
        config=RetrievalConfig(
            top_k=20,
            score_threshold=0.3,
            rerank_enabled=True,
            rerank_model="rerank-v3.5",
            rerank_top_n=5,
            max_pool=80,
        ),
        vectordb_service=vectordb_service,
    )
    rag_service = RAGService(retrieval_service=retrieval_service)

    _pipeline.update({
        "rag": rag_service,
        "retrieval": retrieval_service,
        "vectordb": vectordb_service,
    })
    return rag_service, retrieval_service, vectordb_service


def _build_app() -> TestClient:
    """Wire a FastAPI app with the RAG router, injecting the test services."""
    rag_service, retrieval_service, _ = _build_pipeline()

    import ChatService.Server.routes.rag as rag_mod
    rag_mod._get_rag_service = lambda: rag_service
    rag_mod._get_retrieval_service = lambda: retrieval_service
    rag_mod._get_vectordb_service = lambda: retrieval_service.vectordb_service

    from ChatService.Server.routes.rag import router as rag_router

    app = FastAPI()
    app.include_router(rag_router, prefix="/api")
    return TestClient(app)


_client_cache: dict = {}


def _client() -> TestClient:
    if "client" not in _client_cache:
        _client_cache["client"] = _build_app()
    return _client_cache["client"]


def _contents(chunks) -> str:
    """Lowercased concatenation of chunk contents (SearchResult or dict)."""
    parts = []
    for c in chunks:
        if isinstance(c, dict):
            parts.append(c.get("content", ""))
        else:
            parts.append(c.content)
    return " ".join(parts).lower()


# ════════════════════════════════════════════════════════════════
#  Part A — Service layer: isolation vs. cross-collection visibility
# ════════════════════════════════════════════════════════════════


def test_single_collection_is_isolated():
    """A single collection cannot see content stored in OTHER collections."""
    print("\n" + "=" * 70)
    print("  TEST 1: Single collection is isolated (the core problem)")
    print("=" * 70)

    _, retrieval, _ = _build_pipeline()

    # Docker content lives ONLY in COLL_CONVOS. Searching the resume
    # collection must NOT surface strong Docker content.
    result = asyncio.run(retrieval.aretrieve(
        query="What is Docker and why is it popular?",
        collection_name=COLL_RESUME,
    ))
    content = _contents(result.source_chunks)
    assert "docker" not in content, (
        "Resume collection should not contain Docker content "
        f"(got: {content[:160]})"
    )
    print("    ✅ Resume collection correctly has no Docker content")


def test_retrieve_multi_spans_collections():
    """retrieve_multi pools candidates from every listed collection."""
    print("\n" + "=" * 70)
    print("  TEST 2: Multi-collection retrieval spans collections")
    print("=" * 70)

    _, retrieval, _ = _build_pipeline()

    # Docker (convos) + Microsoft (resume): both should be reachable when
    # we search across both collections.
    result = asyncio.run(retrieval.aretrieve_multi(
        query="What is Docker and why is it popular?",
        collection_names=[COLL_RESUME, COLL_CONVOS],
    ))
    content = _contents(result.source_chunks)
    assert result.source_chunks, "should retrieve chunks"
    assert "docker" in content or "container" in content, (
        f"Multi-collection search should surface Docker content (got: {content[:160]})"
    )
    assert result.collections == [COLL_RESUME, COLL_CONVOS]
    print(f"    ✅ Found Docker content across {len(result.collections)} collections; "
          f"reranked={result.reranked}, pooled={result.total_candidates}")


def test_multi_collection_provenance_tagged():
    """Pooled chunks are tagged with their originating collection."""
    print("\n" + "=" * 70)
    print("  TEST 3: Pooled chunks carry collection provenance")
    print("=" * 70)

    _, retrieval, _ = _build_pipeline()
    result = asyncio.run(retrieval.aretrieve_multi(
        query="Garvit Batra education and Docker containers",
        collection_names=[COLL_RESUME, COLL_CONVOS, COLL_TRAINING],
    ))
    assert result.source_chunks, "should retrieve chunks"
    cols = {c.metadata.get("collection") for c in result.source_chunks}
    assert cols, "chunks should be tagged with a 'collection' field"
    assert cols.issubset({COLL_RESUME, COLL_CONVOS, COLL_TRAINING})
    print(f"    ✅ Chunks tagged with collections: {cols}")


def test_missing_collection_is_skipped():
    """A non-existent collection is skipped, not fatal."""
    print("\n" + "=" * 70)
    print("  TEST 4: Missing collection is skipped gracefully")
    print("=" * 70)

    _, retrieval, _ = _build_pipeline()
    result = asyncio.run(retrieval.aretrieve_multi(
        query="Microsoft internship",
        collection_names=[COLL_RESUME, f"ghost_{RUN_ID}"],
    ))
    content = _contents(result.source_chunks)
    assert result.source_chunks, "real collection should still return results"
    assert any(k in content for k in ["microsoft", "m365", "c#", ".net"])
    print("    ✅ Real collection returned results; ghost collection skipped")


def test_aretrieve_multi_concurrent():
    """Async multi-collection retrieval pools across collections."""
    print("\n" + "=" * 70)
    print("  TEST 5: Async multi-collection retrieval")
    print("=" * 70)

    _, retrieval, _ = _build_pipeline()

    async def _go():
        return await retrieval.aretrieve_multi(
            query="REST versus GraphQL API design",
            collection_names=[COLL_RESUME, COLL_CONVOS],
        )

    result = asyncio.run(_go())
    content = _contents(result.source_chunks)
    assert result.source_chunks, "async multi should retrieve chunks"
    assert any(k in content for k in ["rest", "graphql", "api"])
    print(f"    ✅ Async pooled {result.total_candidates} candidates; "
          f"reranked={result.reranked}")


# ════════════════════════════════════════════════════════════════
#  Part B — HTTP layer: /api/rag/search
# ════════════════════════════════════════════════════════════════


def test_http_search_single_collection_isolated():
    """POST /api/rag/search with one collection cannot see other collections."""
    print("\n" + "=" * 70)
    print("  TEST 6: HTTP search — single collection isolation")
    print("=" * 70)

    r = _client().post("/api/rag/search", json={
        "query": "What is Docker and why is it popular?",
        "collection_name": COLL_RESUME,
    })
    assert r.status_code == 200, f"{r.status_code}: {r.text}"
    data = r.json()
    content = _contents(data["results"])
    assert "docker" not in content, (
        f"Single-collection HTTP search leaked Docker content: {content[:160]}"
    )
    print("    ✅ HTTP single-collection search correctly isolated")


def test_http_search_multi_collection():
    """POST /api/rag/search with collection_names pools across collections."""
    print("\n" + "=" * 70)
    print("  TEST 7: HTTP search — multi-collection (collection_names)")
    print("=" * 70)

    r = _client().post("/api/rag/search", json={
        "query": "What is Docker and why is it popular?",
        "collection_names": [COLL_RESUME, COLL_CONVOS],
    })
    assert r.status_code == 200, f"{r.status_code}: {r.text}"
    data = r.json()
    assert data["results"], "should return pooled results"
    content = _contents(data["results"])
    assert "docker" in content or "container" in content, (
        f"HTTP multi-collection search should surface Docker content: {content[:160]}"
    )
    print(f"    ✅ HTTP multi-collection search found Docker across collections "
          f"({data['total_candidates']} candidates, reranked={data['reranked']})")


def test_http_search_backcompat_single_name():
    """A single-entry collection_names behaves like collection_name (back-compat)."""
    print("\n" + "=" * 70)
    print("  TEST 8: HTTP search — back-compat (single-entry list)")
    print("=" * 70)

    r = _client().post("/api/rag/search", json={
        "query": "Microsoft internship M365",
        "collection_names": [COLL_RESUME],
    })
    assert r.status_code == 200, f"{r.status_code}: {r.text}"
    data = r.json()
    content = _contents(data["results"])
    assert any(k in content for k in ["microsoft", "m365", "c#", ".net"])
    print("    ✅ Single-entry collection_names works like collection_name")


# ════════════════════════════════════════════════════════════════
#  Part C — HTTP layer: /api/rag/chat (full RAG with LLM)
# ════════════════════════════════════════════════════════════════


def test_http_chat_multi_collection_grounded():
    """POST /api/rag/chat spanning collections answers from pooled context."""
    print("\n" + "=" * 70)
    print("  TEST 9: HTTP chat — multi-collection grounded answer")
    print("=" * 70)

    # CGPA (9.55) lives in resume/training; Docker lives in convos. A
    # multi-collection chat should be able to ground a CGPA answer even
    # though we also list an unrelated collection.
    r = _client().post("/api/rag/chat", json={
        "messages": [
            {"role": "user",
             "content": "What was Garvit Batra's CGPA at Delhi Technological University?"}
        ],
        "collection_names": [COLL_CONVOS, COLL_RESUME, COLL_TRAINING],
    })
    assert r.status_code == 200, f"{r.status_code}: {r.text}"
    data = r.json()
    assert data["content"], "answer should not be empty"
    assert "9.55" in data["content"], (
        f"Expected '9.55' in grounded answer, got: {data['content'][:200]}"
    )
    assert data["sources"], "should include source chunks"
    print(f"    ✅ Grounded multi-collection answer contains '9.55' "
          f"({data['provider']}/{data['model']}, {len(data['sources'])} sources)")


def test_http_chat_single_collection_backcompat():
    """POST /api/rag/chat with singular collection_name still works unchanged."""
    print("\n" + "=" * 70)
    print("  TEST 10: HTTP chat — single collection_name (back-compat)")
    print("=" * 70)

    r = _client().post("/api/rag/chat", json={
        "messages": [
            {"role": "user", "content": "Where did Garvit intern and on what team?"}
        ],
        "collection_name": COLL_RESUME,
    })
    assert r.status_code == 200, f"{r.status_code}: {r.text}"
    data = r.json()
    assert data["content"], "answer should not be empty"
    assert "microsoft" in data["content"].lower(), (
        f"Expected Microsoft in answer, got: {data['content'][:200]}"
    )
    print(f"    ✅ Single-collection chat still works "
          f"({data['provider']}/{data['model']})")


# ════════════════════════════════════════════════════════════════
#  Runner
# ════════════════════════════════════════════════════════════════


def main():
    if not os.getenv("COHERE_API_KEY"):
        print("❌  COHERE_API_KEY not set — cannot run multi-collection E2E tests.")
        sys.exit(1)

    print(f"\n{'=' * 70}")
    print(f"  Multi-Collection RAG E2E Suite   (run {RUN_ID})")
    print(f"{'=' * 70}")
    print("  Building 3-collection in-memory pipeline...")
    _build_pipeline()

    tests = [
        # Part A — service layer
        test_single_collection_is_isolated,
        test_retrieve_multi_spans_collections,
        test_multi_collection_provenance_tagged,
        test_missing_collection_is_skipped,
        test_aretrieve_multi_concurrent,
        # Part B — HTTP /search
        test_http_search_single_collection_isolated,
        test_http_search_multi_collection,
        test_http_search_backcompat_single_name,
        # Part C — HTTP /chat (LLM)
        test_http_chat_multi_collection_grounded,
        test_http_chat_single_collection_backcompat,
    ]

    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            failed += 1
            print(f"    ❌ ASSERTION FAILED: {e}")
            traceback.print_exc()
        except Exception as e:
            failed += 1
            print(f"    ❌ ERROR: {e}")
            traceback.print_exc()

    print(f"\n{'=' * 70}")
    print(f"  Results: {passed} passed, {failed} failed / {len(tests)} total")
    print(f"{'=' * 70}\n")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
