"""
End-to-end test for streaming RAG metadata (model / provider / token usage).

Validates that POST /api/rag/chat/stream emits, in its final ``done`` SSE
event, the same identifying fields the non-streaming /api/rag/chat returns:
    - provider   (which LLM provider served the request)
    - model      (which model generated the response)
    - usage      (prompt_tokens / completion_tokens / total_tokens)

Uses an in-memory Qdrant collection + real Cohere (embeddings + the default
RAG LLM, command-a-03-2025, which emits usage on its stream's message-end
event), wired through the real FastAPI RAG router via TestClient.

Requires (.env):
    COHERE_API_KEY  — Cohere embed-english-v3.0 + command-a-03-2025

Run:
    PYTHONPATH=. python tests/test_rag_stream_metadata_e2e.py
    PYTHONPATH=. python -m pytest tests/test_rag_stream_metadata_e2e.py -v --tb=short
"""

from __future__ import annotations

import asyncio
import json
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

EMBEDDING_DIM = 1024
RUN_ID = uuid.uuid4().hex[:8]

DATA_DIR = Path(__file__).parent / "document_loaders_tests"
TXT_FILE = DATA_DIR / "my all details.txt"
COLLECTION = f"stream_meta_{RUN_ID}"

_pipeline: dict = {}


def _build_pipeline():
    if _pipeline:
        return _pipeline["rag"], _pipeline["retrieval"]

    embeddings = EmbeddingsFactory.create_cohere(model_name="embed-english-v3.0")
    vectordb = VectorDBFactory.create_from_env(
        provider=VectorDBProvider.QDRANT,
        collection_name=COLLECTION,
        embedding_dimension=EMBEDDING_DIM,
        in_memory=True,
        distance_metric=DistanceMetric.COSINE,
    )
    vectordb_service = VectorDBService(
        config=VectorDBServiceConfig(
            collection_name=COLLECTION,
            in_memory=True,
            auto_create_collection=True,
        ),
        vectordb=vectordb,
        embeddings=embeddings,
    )
    upload_svc = AssetUploadService(
        config=AssetUploadConfig(
            default_collection=COLLECTION,
            chunk_size=1000,
            chunk_overlap=200,
            use_smart_chunker=True,
            use_llm_analysis=False,
        ),
        vectordb=vectordb,
        embeddings=embeddings,
    )
    # Seed via the async path so the data lands in the async-client store the
    # streaming endpoint queries (in-memory Qdrant keeps separate sync/async
    # stores; production shares one server).
    assert TXT_FILE.exists(), f"Missing test data file: {TXT_FILE}"
    result = asyncio.run(
        upload_svc.async_upload_file(TXT_FILE, collection_name=COLLECTION)
    )
    assert result.success, f"Upload failed: {result.error}"

    retrieval_service = RetrievalService(
        config=RetrievalConfig(
            top_k=20, score_threshold=0.3, rerank_enabled=True, rerank_top_n=5,
        ),
        vectordb_service=vectordb_service,
    )
    rag_service = RAGService(retrieval_service=retrieval_service)

    _pipeline.update({"rag": rag_service, "retrieval": retrieval_service})
    return rag_service, retrieval_service


def _client() -> TestClient:
    rag_service, retrieval_service = _build_pipeline()

    import ChatService.Server.routes.rag as rag_mod
    rag_mod._get_rag_service = lambda: rag_service
    rag_mod._get_retrieval_service = lambda: retrieval_service
    rag_mod._get_vectordb_service = lambda: retrieval_service.vectordb_service

    from ChatService.Server.routes.rag import router as rag_router

    app = FastAPI()
    app.include_router(rag_router, prefix="/api")
    return TestClient(app)


def _parse_sse(raw: str) -> list[dict]:
    """Parse an SSE response body into a list of JSON event objects."""
    events = []
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            payload = line[len("data:"):].strip()
            if payload:
                events.append(json.loads(payload))
    return events


def test_stream_done_event_has_model_provider_usage():
    """The streaming done event carries provider, model, and token usage."""
    print("\n" + "=" * 70)
    print("  TEST: stream done event includes model / provider / usage")
    print("=" * 70)

    client = _client()
    resp = client.post("/api/rag/chat/stream", json={
        "messages": [
            {"role": "user",
             "content": "Where did Garvit intern and on what team?"}
        ],
        "collection_name": COLLECTION,
    })
    assert resp.status_code == 200, f"{resp.status_code}: {resp.text}"

    events = _parse_sse(resp.text)
    assert events, "expected SSE events"

    # Content must have streamed
    content = "".join(
        e.get("content", "") for e in events if e.get("type") == "content"
    )
    assert content.strip(), "should have streamed some content"

    # Exactly one done event, carrying the metadata
    done = [e for e in events if e.get("type") == "done"]
    assert len(done) == 1, f"expected one done event, got {len(done)}"
    done = done[0]

    assert done["done"] is True
    assert done.get("provider"), f"done event missing provider: {done}"
    assert done.get("model"), f"done event missing model: {done}"

    usage = done.get("usage")
    assert usage is not None, f"done event missing usage: {done}"
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        assert key in usage, f"usage missing {key}: {usage}"
    assert usage["total_tokens"] > 0, f"expected positive total_tokens: {usage}"
    assert usage["prompt_tokens"] > 0, f"expected positive prompt_tokens: {usage}"
    assert usage["completion_tokens"] > 0, f"expected positive completion_tokens: {usage}"

    print(f"    ✅ provider={done['provider']} model={done['model']}")
    print(f"    ✅ usage={usage}")
    print(f"    Answer preview: {content[:120]}...")


def test_stream_sources_event_precedes_content():
    """Sanity: sources event arrives before content, done is last."""
    print("\n" + "=" * 70)
    print("  TEST: SSE event ordering (sources → content → done)")
    print("=" * 70)

    client = _client()
    resp = client.post("/api/rag/chat/stream", json={
        "messages": [{"role": "user", "content": "What is Garvit's CGPA?"}],
        "collection_name": COLLECTION,
    })
    assert resp.status_code == 200, f"{resp.status_code}: {resp.text}"
    events = _parse_sse(resp.text)
    types = [e.get("type") for e in events]

    assert types[0] == "sources", f"first event should be sources, got {types[:3]}"
    assert types[-1] == "done", f"last event should be done, got {types[-3:]}"
    assert "content" in types, "should contain content events"
    print(f"    ✅ event order ok: sources … content … done ({len(events)} events)")


def main():
    if not os.getenv("COHERE_API_KEY"):
        print("❌  COHERE_API_KEY not set — cannot run streaming metadata E2E.")
        sys.exit(1)

    print(f"\n{'=' * 70}")
    print(f"  RAG Streaming Metadata E2E   (run {RUN_ID})")
    print(f"{'=' * 70}")

    tests = [
        test_stream_done_event_has_model_provider_usage,
        test_stream_sources_event_precedes_content,
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
