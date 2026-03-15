"""
End-to-end tests for the RAG retrieval + generation pipeline.

Tests the full production workflow:
    Upload documents → Vector search → Cohere Rerank → LLM generation

Covers TWO modes:
    Part A (1–6) : In-memory Qdrant — fast, no cloud credentials needed
    Part B (7–8) : In-memory Reranker integration
    Part C (9–13): Qdrant Cloud — production-like, real persistence

Data: Uses the real files from tests/document_loaders_tests/:
    - my all details.txt        (Garvit Batra's resume)
    - Personal_TrainingData.csv (prompt/response pairs about Garvit)
    - conversation.json         (tech Q&A conversations)

Run:
    PYTHONPATH=. python tests/test_rag_e2e.py

    # Cloud tests only run when Qdrant Cloud credentials are set:
    QDRANT_URL=... QDRANT_API_KEY=... PYTHONPATH=. python tests/test_rag_e2e.py

    # Keep cloud collection for dashboard inspection:
    PYTHONPATH=. python tests/test_rag_e2e.py --persist

Requires (.env):
    COHERE_API_KEY     – Cohere embed-english-v3.0, rerank-v3.5, command-a-03-2025
    QDRANT_URL         – Qdrant Cloud cluster URL    (for Part C)
    QDRANT_API_KEY     – Qdrant Cloud API key        (for Part C)
"""

from __future__ import annotations

import os
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# ── Imports ──────────────────────────────────────────────────────

from RAGService.Data.VectorDB import (
    BaseVectorDB,
    DocumentChunk,
    VectorDBFactory,
    VectorDBProvider,
    DistanceMetric,
    SearchResult,
)
from RAGService.Data.Embeddings import (
    EmbeddingsFactory,
    EmbeddingConfig,
    EmbeddingProvider,
    EmbeddingInputType,
)
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
    RetrievalResult,
)
from RAGService.Data.services.rag_service import (
    RAGService,
    RAGConfig,
    RAGResponse,
)
from ChatService.Chat.llm.base import LLMProvider


# ── Configuration ────────────────────────────────────────────────

QDRANT_URL = os.getenv("QDRANT_URL")
EMBEDDING_DIM = 1024  # embed-english-v3.0

RUN_ID = uuid.uuid4().hex[:8]

# Test data directory
DATA_DIR = Path(__file__).parent / "document_loaders_tests"
TXT_FILE = DATA_DIR / "my all details.txt"
CSV_FILE = DATA_DIR / "Personal_TrainingData.csv"
JSON_FILE = DATA_DIR / "conversation.json"


# ── Shared Helpers ───────────────────────────────────────────────

def _build_embeddings():
    """Build a real Cohere embeddings instance."""
    return EmbeddingsFactory.create_cohere(model_name="embed-english-v3.0")


def _populate_collection(
    upload_svc: AssetUploadService,
    collection: str,
) -> int:
    """
    Upload all 3 test data files into the given collection.
    Returns total chunk count.
    """
    total_chunks = 0
    for label, fpath in [("TXT", TXT_FILE), ("CSV", CSV_FILE), ("JSON", JSON_FILE)]:
        if not fpath.exists():
            print(f"    ⚠️  Skipping {label}: {fpath} not found")
            continue
        result = upload_svc.upload_file(fpath, collection_name=collection)
        assert result.success, f"Upload failed for {label}: {result.error}"
        total_chunks += result.total_chunks
        print(f"    ✅ {label}: {result.total_chunks} chunks uploaded")

    return total_chunks


def _wait(seconds: float = 1.5):
    """Pause for Qdrant Cloud eventual consistency."""
    time.sleep(seconds)


def _safe_cleanup(vectordb: BaseVectorDB, collection: str):
    """Silently delete a cloud collection."""
    try:
        vectordb.delete_collection(collection)
    except Exception:
        pass


# ── In-Memory Pipeline Builder ───────────────────────────────────

_mem_pipeline_cache = {}


def _get_inmemory_pipeline():
    """
    Build (or reuse) a fully populated in-memory RAG pipeline.

    Cached so all Part A/B tests share one embedding call
    (Cohere embed is the slowest part — avoids duplicate API costs).
    """
    if _mem_pipeline_cache:
        return (
            _mem_pipeline_cache["rag_service"],
            _mem_pipeline_cache["retrieval_service"],
            _mem_pipeline_cache["vectordb_service"],
            _mem_pipeline_cache["collection"],
        )

    collection = f"rag_mem_{RUN_ID}"
    print(f"\n  🔧 Building in-memory RAG pipeline (collection: {collection})")

    embeddings = _build_embeddings()

    vectordb = VectorDBFactory.create_from_env(
        provider=VectorDBProvider.QDRANT,
        collection_name=collection,
        embedding_dimension=EMBEDDING_DIM,
        in_memory=True,
        distance_metric=DistanceMetric.COSINE,
    )

    # VectorDBService for retrieval (shared embeddings + vectordb)
    vectordb_config = VectorDBServiceConfig(
        collection_name=collection,
        in_memory=True,
        auto_create_collection=True,
    )
    vectordb_service = VectorDBService(
        config=vectordb_config,
        vectordb=vectordb,
        embeddings=embeddings,
    )

    # Upload data using AssetUploadService
    upload_config = AssetUploadConfig(
        default_collection=collection,
        chunk_size=1000,
        chunk_overlap=200,
        use_smart_chunker=True,
        use_llm_analysis=False,
    )
    upload_svc = AssetUploadService(
        config=upload_config,
        vectordb=vectordb,
        embeddings=embeddings,
    )

    print(f"  📄 Uploading test data files...")
    total = _populate_collection(upload_svc, collection)
    count = vectordb_service.count(collection_name=collection)
    print(f"  📊 Collection '{collection}': {count} vectors (from {total} chunks)")

    # Build RetrievalService with reranking enabled
    retrieval_config = RetrievalConfig(
        top_k=20,
        score_threshold=0.3,
        rerank_enabled=True,
        rerank_model="rerank-v3.5",
        rerank_top_n=5,
    )
    retrieval_service = RetrievalService(
        config=retrieval_config,
        vectordb_service=vectordb_service,
    )

    # Build RAGService with the pre-built RetrievalService
    rag_service = RAGService(retrieval_service=retrieval_service)

    _mem_pipeline_cache.update({
        "rag_service": rag_service,
        "retrieval_service": retrieval_service,
        "vectordb_service": vectordb_service,
        "collection": collection,
    })

    return rag_service, retrieval_service, vectordb_service, collection


# ── Cloud Pipeline Builder ───────────────────────────────────────

_cloud_pipeline_cache = {}


def _get_cloud_pipeline():
    """
    Build (or reuse) a fully populated cloud Qdrant RAG pipeline.

    Uses prefer_grpc=False (REST) since gRPC hangs on some Qdrant Cloud setups.
    """
    if _cloud_pipeline_cache:
        return (
            _cloud_pipeline_cache["rag_service"],
            _cloud_pipeline_cache["retrieval_service"],
            _cloud_pipeline_cache["vectordb_service"],
            _cloud_pipeline_cache["collection"],
        )

    collection = f"rag_cloud_{RUN_ID}"
    print(f"\n  🔧 Building cloud RAG pipeline (collection: {collection})")
    print(f"      Qdrant URL: {QDRANT_URL}")

    embeddings = _build_embeddings()

    vectordb = VectorDBFactory.create_from_env(
        provider=VectorDBProvider.QDRANT,
        collection_name=collection,
        embedding_dimension=EMBEDDING_DIM,
        distance_metric=DistanceMetric.COSINE,
        prefer_grpc=False,
    )

    vectordb_config = VectorDBServiceConfig(
        collection_name=collection,
        auto_create_collection=True,
    )
    vectordb_service = VectorDBService(
        config=vectordb_config,
        vectordb=vectordb,
        embeddings=embeddings,
    )

    upload_config = AssetUploadConfig(
        default_collection=collection,
        chunk_size=1000,
        chunk_overlap=200,
        use_smart_chunker=True,
        use_llm_analysis=False,
    )
    upload_svc = AssetUploadService(
        config=upload_config,
        vectordb=vectordb,
        embeddings=embeddings,
    )

    print(f"  📄 Uploading test data files to cloud...")
    total = _populate_collection(upload_svc, collection)
    _wait(2.0)  # cloud eventual consistency
    count = vectordb_service.count(collection_name=collection)
    print(f"  📊 Cloud collection '{collection}': {count} vectors (from {total} chunks)")

    retrieval_config = RetrievalConfig(
        top_k=20,
        score_threshold=0.3,
        rerank_enabled=True,
        rerank_model="rerank-v3.5",
        rerank_top_n=5,
    )
    retrieval_service = RetrievalService(
        config=retrieval_config,
        vectordb_service=vectordb_service,
    )

    rag_service = RAGService(retrieval_service=retrieval_service)

    _cloud_pipeline_cache.update({
        "rag_service": rag_service,
        "retrieval_service": retrieval_service,
        "vectordb_service": vectordb_service,
        "collection": collection,
    })

    return rag_service, retrieval_service, vectordb_service, collection


# ════════════════════════════════════════════════════════════════
#  Part A: In-Memory Retrieval Tests (no LLM)
# ════════════════════════════════════════════════════════════════

def test_mem_retrieval_microsoft_internship():
    """Retrieve context about Garvit's Microsoft internship from the TXT resume."""
    _, retrieval_service, _, collection = _get_inmemory_pipeline()

    result = retrieval_service.retrieve(
        query="Where did Garvit intern at Microsoft and what did he work on?",
        collection_name=collection,
    )

    assert result.context_str, "context_str should not be empty"
    assert result.source_chunks, "source_chunks should not be empty"
    assert result.total_candidates > 0, "should have candidates from vector search"

    # Check that retrieved chunks contain Microsoft-related content
    all_content = " ".join(c.content for c in result.source_chunks).lower()
    assert any(
        kw in all_content for kw in ["microsoft", "m365", "c#", ".net"]
    ), f"Expected Microsoft-related content in chunks, got: {all_content[:200]}"

    print(f"    ✅ Retrieved {len(result.source_chunks)} chunks, "
          f"reranked={result.reranked}, candidates={result.total_candidates}")
    print(f"    Top chunk (score={result.source_chunks[0].score:.4f}): "
          f"{result.source_chunks[0].content[:100]}...")


def test_mem_retrieval_docker_conversation():
    """Retrieve context about Docker from the conversation.json data."""
    _, retrieval_service, _, collection = _get_inmemory_pipeline()

    result = retrieval_service.retrieve(
        query="What is Docker and why is it so popular in modern development?",
        collection_name=collection,
    )

    assert result.context_str, "context_str should not be empty"
    assert result.source_chunks, "source_chunks should not be empty"

    all_content = " ".join(c.content for c in result.source_chunks).lower()
    assert any(
        kw in all_content for kw in ["docker", "container", "environment"]
    ), f"Expected Docker-related content, got: {all_content[:200]}"

    print(f"    ✅ Retrieved {len(result.source_chunks)} chunks about Docker")
    print(f"    Top chunk (score={result.source_chunks[0].score:.4f}): "
          f"{result.source_chunks[0].content[:100]}...")


def test_mem_retrieval_findone_project():
    """Retrieve context about the FindOne face-recognition project."""
    _, retrieval_service, _, collection = _get_inmemory_pipeline()

    result = retrieval_service.retrieve(
        query="FindOne project face recognition technology stack",
        collection_name=collection,
    )

    assert result.source_chunks, "should have source chunks"

    all_content = " ".join(c.content for c in result.source_chunks).lower()
    assert any(
        kw in all_content for kw in ["findone", "face recognition", "face-recognition", "opencv"]
    ), f"Expected FindOne-related content, got: {all_content[:200]}"

    print(f"    ✅ Retrieved {len(result.source_chunks)} chunks about FindOne")


def test_mem_search_only_competitive_programming():
    """Use search_only (no rerank, no formatting) for competitive programming data."""
    _, retrieval_service, _, collection = _get_inmemory_pipeline()

    results = retrieval_service.search_only(
        query="competitive programming Codeforces rating achievements",
        collection_name=collection,
        k=10,
    )

    assert isinstance(results, list), "search_only should return a list"
    assert len(results) > 0, "should find competitive programming content"

    all_content = " ".join(r.content for r in results).lower()
    assert any(
        kw in all_content for kw in ["codeforces", "1414", "codechef", "1773"]
    ), f"Expected competitive programming content, got: {all_content[:200]}"

    # search_only results should NOT have rerank metadata
    for r in results:
        assert r.metadata.get("reranked") is not True, (
            "search_only should bypass reranker"
        )

    print(f"    ✅ search_only returned {len(results)} raw results")
    print(f"    Top result (score={results[0].score:.4f}): "
          f"{results[0].content[:100]}...")


def test_mem_retrieval_csv_personal_data():
    """Retrieve context from the CSV training data about Garvit's education."""
    _, retrieval_service, _, collection = _get_inmemory_pipeline()

    result = retrieval_service.retrieve(
        query="Tell me about Garvit Batra's education background and CGPA",
        collection_name=collection,
    )

    assert result.source_chunks, "should have source chunks"

    all_content = " ".join(c.content for c in result.source_chunks).lower()
    assert any(
        kw in all_content for kw in ["9.55", "dtu", "delhi technological", "b.tech"]
    ), f"Expected education content, got: {all_content[:200]}"

    print(f"    ✅ Retrieved {len(result.source_chunks)} chunks about education")


# ════════════════════════════════════════════════════════════════
#  Part B: In-Memory Reranker Integration Tests
# ════════════════════════════════════════════════════════════════

def test_mem_rerank_changes_ordering():
    """Verify that reranking modifies result scores and adds rerank metadata."""
    _, reranked_retrieval, vectordb_service, collection = _get_inmemory_pipeline()

    # Build a second retrieval service with reranking DISABLED
    no_rerank_config = RetrievalConfig(
        top_k=20,
        score_threshold=0.3,
        rerank_enabled=False,
    )
    no_rerank_retrieval = RetrievalService(
        config=no_rerank_config,
        vectordb_service=vectordb_service,
    )

    query = "What projects has Garvit built using React and Node?"

    # Retrieve WITH reranking (from cached pipeline)
    reranked_result = reranked_retrieval.retrieve(
        query=query, collection_name=collection,
    )

    # Retrieve WITHOUT reranking
    plain_result = no_rerank_retrieval.retrieve(
        query=query, collection_name=collection,
    )

    assert reranked_result.reranked is True, "Should be reranked"
    assert plain_result.reranked is False, "Should NOT be reranked"

    # Reranked results should have rerank metadata
    for chunk in reranked_result.source_chunks:
        assert chunk.metadata.get("reranked") is True, "Missing reranked flag"
        assert "rerank_score" in chunk.metadata, "Missing rerank_score"
        assert "original_score" in chunk.metadata, "Missing original_score"

    # Plain results should NOT have rerank metadata
    for chunk in plain_result.source_chunks:
        assert chunk.metadata.get("reranked") is not True, (
            "Plain results shouldn't be reranked"
        )

    print(f"    ✅ Reranked: {len(reranked_result.source_chunks)} chunks "
          f"(top score: {reranked_result.source_chunks[0].score:.4f})")
    print(f"    ✅ Plain:    {len(plain_result.source_chunks)} chunks "
          f"(top score: {plain_result.source_chunks[0].score:.4f})")
    print(f"    Reranked result IDs:  {[c.id[:8] for c in reranked_result.source_chunks[:3]]}")
    print(f"    Plain result IDs:     {[c.id[:8] for c in plain_result.source_chunks[:3]]}")


def test_mem_rerank_score_populated():
    """Verify reranked results have valid rerank_score between 0 and 1."""
    _, retrieval_service, _, collection = _get_inmemory_pipeline()

    result = retrieval_service.retrieve(
        query="What internships has Garvit Batra done?",
        collection_name=collection,
    )

    assert result.reranked, "Should be reranked"
    for chunk in result.source_chunks:
        score = chunk.metadata.get("rerank_score")
        assert score is not None, "rerank_score should be present"
        assert 0.0 <= score <= 1.0, f"rerank_score should be 0–1, got {score}"

    print(f"    ✅ All {len(result.source_chunks)} chunks have valid rerank_score")


# ════════════════════════════════════════════════════════════════
#  Part A+: In-Memory Full RAG Pipeline Tests (with LLM)
# ════════════════════════════════════════════════════════════════

def test_mem_rag_query_projects():
    """Full RAG query: retrieve + rerank + LLM generation about projects."""
    rag_service, _, _, collection = _get_inmemory_pipeline()

    response = rag_service.query(
        user_query="What projects has Garvit Batra built? List them with their tech stacks.",
        collection_name=collection,
    )

    assert isinstance(response, RAGResponse)
    assert response.answer, "answer should not be empty"
    assert len(response.answer) > 50, "answer should be substantive"
    assert response.sources, "should have source chunks"
    assert response.provider_used, "provider_used should be set"
    assert response.model_used, "model_used should be set"
    assert response.retrieval_result.source_chunks, "retrieval should have chunks"

    print(f"    ✅ RAG answer generated ({len(response.answer)} chars)")
    print(f"    Provider: {response.provider_used}/{response.model_used}")
    print(f"    Sources: {len(response.sources)} chunks")
    print(f"    Answer preview: {response.answer[:150]}...")


def test_mem_rag_query_factual_cgpa():
    """Full RAG query with factual grounding check — answer must contain '9.55'."""
    rag_service, _, _, collection = _get_inmemory_pipeline()

    response = rag_service.query(
        user_query="What was Garvit Batra's CGPA at DTU?",
        collection_name=collection,
    )

    assert response.answer, "answer should not be empty"
    assert "9.55" in response.answer, (
        f"Expected '9.55' in answer for factual grounding, got: {response.answer[:200]}"
    )

    print(f"    ✅ Factual grounding passed — '9.55' found in answer")
    print(f"    Answer: {response.answer[:150]}...")


def test_mem_rag_query_stream():
    """Full RAG streaming query — collect all chunks and verify."""
    rag_service, _, _, collection = _get_inmemory_pipeline()

    retrieval_result, stream = rag_service.query_stream(
        user_query="Explain the difference between REST and GraphQL",
        collection_name=collection,
    )

    assert retrieval_result.source_chunks, "retrieval should have chunks"

    # Collect stream
    chunks = []
    for chunk in stream:
        chunks.append(chunk)

    full_answer = "".join(chunks)
    assert full_answer, "streamed answer should not be empty"
    assert len(chunks) > 1, "should have multiple stream chunks"

    print(f"    ✅ Streamed {len(chunks)} chunks ({len(full_answer)} chars total)")
    print(f"    Sources: {len(retrieval_result.source_chunks)} chunks "
          f"(reranked={retrieval_result.reranked})")
    print(f"    Answer preview: {full_answer[:150]}...")


# ════════════════════════════════════════════════════════════════
#  Part C: Cloud Qdrant Tests
# ════════════════════════════════════════════════════════════════

def test_cloud_retrieval_internship():
    """Cloud: retrieve context about Microsoft internship."""
    _, retrieval_service, _, collection = _get_cloud_pipeline()

    result = retrieval_service.retrieve(
        query="Where did Garvit intern at Microsoft and what did he work on?",
        collection_name=collection,
    )

    assert result.context_str, "context_str should not be empty"
    assert result.source_chunks, "source_chunks should not be empty"

    all_content = " ".join(c.content for c in result.source_chunks).lower()
    assert any(
        kw in all_content for kw in ["microsoft", "m365", "c#", ".net"]
    ), f"Expected Microsoft content, got: {all_content[:200]}"

    print(f"    ✅ Cloud retrieved {len(result.source_chunks)} chunks about Microsoft")


def test_cloud_retrieval_csv_contact():
    """Cloud: retrieve contact info from CSV data."""
    _, retrieval_service, _, collection = _get_cloud_pipeline()

    result = retrieval_service.retrieve(
        query="What is Garvit Batra's phone number and email address?",
        collection_name=collection,
    )

    assert result.source_chunks, "should have chunks"

    all_content = " ".join(c.content for c in result.source_chunks).lower()
    assert any(
        kw in all_content for kw in ["9310502497", "gbatra145", "garvitbatra"]
    ), f"Expected contact info, got: {all_content[:200]}"

    print(f"    ✅ Cloud retrieved contact info from CSV data")


def test_cloud_rag_query_factual_cgpa():
    """Cloud: full RAG query with factual grounding — answer must contain '9.55'."""
    rag_service, _, _, collection = _get_cloud_pipeline()

    response = rag_service.query(
        user_query="What was Garvit Batra's CGPA at Delhi Technological University?",
        collection_name=collection,
    )

    assert response.answer, "answer should not be empty"
    assert "9.55" in response.answer, (
        f"Expected '9.55' in answer, got: {response.answer[:200]}"
    )

    print(f"    ✅ Cloud factual grounding passed — '9.55' in answer")
    print(f"    Provider: {response.provider_used}/{response.model_used}")


def test_cloud_rag_query_rest_graphql():
    """Cloud: full RAG query about REST vs GraphQL from conversation data."""
    rag_service, _, _, collection = _get_cloud_pipeline()

    response = rag_service.query(
        user_query="Explain the difference between REST and GraphQL APIs",
        collection_name=collection,
    )

    assert response.answer, "answer should not be empty"
    assert response.sources, "should have source chunks"

    # The answer should reference concepts from the conversation.json data
    answer_lower = response.answer.lower()
    assert any(
        kw in answer_lower for kw in ["rest", "graphql", "api", "fetch"]
    ), f"Expected REST/GraphQL content, got: {response.answer[:200]}"

    print(f"    ✅ Cloud RAG answered REST vs GraphQL question")
    print(f"    Answer preview: {response.answer[:150]}...")


def test_cloud_rag_query_stream():
    """Cloud: full RAG streaming query."""
    rag_service, _, _, collection = _get_cloud_pipeline()

    retrieval_result, stream = rag_service.query_stream(
        user_query="What are Garvit Batra's technical skills and what languages does he know?",
        collection_name=collection,
    )

    assert retrieval_result.source_chunks, "retrieval should have chunks"

    chunks = []
    for chunk in stream:
        chunks.append(chunk)

    full_answer = "".join(chunks)
    assert full_answer, "streamed answer should not be empty"
    assert len(chunks) > 1, "should have multiple stream chunks"

    print(f"    ✅ Cloud streamed {len(chunks)} chunks ({len(full_answer)} chars)")
    print(f"    Answer preview: {full_answer[:150]}...")


# ════════════════════════════════════════════════════════════════
#  Runner
# ════════════════════════════════════════════════════════════════

def main():
    """Run all E2E RAG tests with pass/fail/skip tracking."""
    persist = "--persist" in sys.argv

    has_cohere = bool(os.getenv("COHERE_API_KEY"))
    has_cloud = bool(QDRANT_URL and os.getenv("QDRANT_API_KEY"))

    if not has_cohere:
        print("❌  COHERE_API_KEY not set — cannot run any RAG tests.")
        sys.exit(1)

    print(f"\n{'=' * 70}")
    print(f"  RAG E2E Test Suite")
    print(f"  Run ID     : {RUN_ID}")
    print(f"  Cohere key : {'✅ set' if has_cohere else '❌ missing'}")
    print(f"  Qdrant URL : {QDRANT_URL or '❌ not set (cloud tests will be skipped)'}")
    print(f"  Persist    : {'yes' if persist else 'no'}")
    print(f"{'=' * 70}\n")

    passed = 0
    failed = 0
    skipped = 0

    def _run(test_fn, requires_cloud=False, requires_llm=False):
        nonlocal passed, failed, skipped

        if requires_cloud and not has_cloud:
            print(f"  ⏭️  SKIPPED  {test_fn.__name__} (no Qdrant Cloud credentials)")
            skipped += 1
            return

        print(f"\n  🧪 {test_fn.__name__}")
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            failed += 1
            print(f"    ❌ ASSERTION FAILED: {e}")
            traceback.print_exc()
        except Exception as e:
            failed += 1
            print(f"    ❌ ERROR: {e}")
            traceback.print_exc()

    # ── Part A: In-Memory Retrieval (no LLM) ────────────────────
    print(f"\n{'─' * 70}")
    print(f"  Part A: In-Memory Retrieval Tests")
    print(f"{'─' * 70}")

    _run(test_mem_retrieval_microsoft_internship)
    _run(test_mem_retrieval_docker_conversation)
    _run(test_mem_retrieval_findone_project)
    _run(test_mem_search_only_competitive_programming)
    _run(test_mem_retrieval_csv_personal_data)

    # ── Part B: In-Memory Reranker Integration ───────────────────
    print(f"\n{'─' * 70}")
    print(f"  Part B: In-Memory Reranker Integration Tests")
    print(f"{'─' * 70}")

    _run(test_mem_rerank_changes_ordering)
    _run(test_mem_rerank_score_populated)

    # ── Part A+: In-Memory Full RAG (with LLM) ──────────────────
    print(f"\n{'─' * 70}")
    print(f"  Part A+: In-Memory Full RAG Pipeline Tests (with LLM)")
    print(f"{'─' * 70}")

    _run(test_mem_rag_query_projects, requires_llm=True)
    _run(test_mem_rag_query_factual_cgpa, requires_llm=True)
    _run(test_mem_rag_query_stream, requires_llm=True)

    # ── Part C: Cloud Qdrant Tests ───────────────────────────────
    print(f"\n{'─' * 70}")
    print(f"  Part C: Cloud Qdrant Tests")
    print(f"{'─' * 70}")

    try:
        _run(test_cloud_retrieval_internship, requires_cloud=True)
        _run(test_cloud_retrieval_csv_contact, requires_cloud=True)
        _run(test_cloud_rag_query_factual_cgpa, requires_cloud=True)
        _run(test_cloud_rag_query_rest_graphql, requires_cloud=True)
        _run(test_cloud_rag_query_stream, requires_cloud=True)
    finally:
        # Cleanup cloud collection unless --persist
        if has_cloud and _cloud_pipeline_cache:
            if persist:
                coll = _cloud_pipeline_cache["collection"]
                print(f"\n  📌 Cloud collection '{coll}' preserved (--persist mode)")
                print(f"     → Check your Qdrant dashboard to inspect the data")
            else:
                coll = _cloud_pipeline_cache["collection"]
                vectordb = _cloud_pipeline_cache["vectordb_service"].vectordb
                print(f"\n  🗑️  Cleaning up cloud collection '{coll}'...")
                _safe_cleanup(vectordb, coll)

    # ── Summary ──────────────────────────────────────────────────
    total = passed + failed + skipped
    print(f"\n{'=' * 70}")
    print(f"  RAG E2E Test Results: {passed} passed, {failed} failed, {skipped} skipped / {total} total")

    if failed == 0:
        print(f"  ✅ All {passed} tests passed!" +
              (f" ({skipped} skipped)" if skipped else ""))
    else:
        print(f"  ❌ {failed} test(s) failed")

    print(f"{'=' * 70}\n")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
