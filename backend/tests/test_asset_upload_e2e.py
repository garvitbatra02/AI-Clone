"""
End-to-end tests for the AssetUploadService.

The "uber test": pick a file → chunk it → inspect every chunk → embed →
upload to VectorDB → retrieve & verify embeddings.

Covers FOUR axes in a matrix:
    File formats : TXT, CSV, JSON, PDF, raw text, directory upload
    Chunking     : rule-based (no LLM) vs LLM-assisted (Groq)
    VectorDB     : in-memory Qdrant vs Qdrant Cloud
    Retrieval    : get_by_ids (with embeddings) + similarity search

Tests are grouped into 4 parts:
    Part A (1-6)  : In-memory  — TXT, CSV, JSON, raw text, directory, search
    Part B (7-9)  : LLM-assisted chunking — TXT topical, PDF structural, comparison
    Part C (10-13): Qdrant Cloud — TXT, CSV, JSON, search-after-upload
    Part D (14)   : LLM + Cloud combined

Run:
    PYTHONPATH=. python tests/test_asset_upload_e2e.py

Requires (.env):
    COHERE_API_KEY     – Cohere embed-english-v3.0
    GROQ_API_KEYS      – Groq llama-3.1-8b-instant  (for Part B / D)
    QDRANT_URL         – Qdrant Cloud cluster URL    (for Part C / D)
    QDRANT_API_KEY     – Qdrant Cloud API key        (for Part C / D)
"""

from __future__ import annotations

import os
import sys
import time
import uuid
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()

# ── Imports ──────────────────────────────────────────────────────

from RAGService.Data.VectorDB import (
    BaseVectorDB,
    DocumentChunk,
    VectorDBConfig,
    VectorDBFactory,
    VectorDBProvider,
    DistanceMetric,
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
    UploadResult,
)

# ── Configuration ────────────────────────────────────────────────

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

EMBEDDING_DIM = 1024  # embed-english-v3.0

RUN_ID = uuid.uuid4().hex[:8]

# Test data directory
DATA_DIR = Path(__file__).parent / "document_loaders_tests"
TXT_FILE = DATA_DIR / "my all details.txt"
CSV_FILE = DATA_DIR / "Personal_TrainingData.csv"
JSON_FILE = DATA_DIR / "conversation.json"
PDF_FILE = DATA_DIR / "8bf0e5bb-8dcf-4b75-b3c1-44e337e6a4ff.pdf"


def _get_groq_keys() -> list[str]:
    multi = os.environ.get("GROQ_API_KEYS", "")
    if multi:
        return [k.strip() for k in multi.split(",") if k.strip()]
    single = os.environ.get("GROQ_API_KEY", "")
    if single:
        return [single.strip()]
    return []


# ── Helper: pretty-print chunks ─────────────────────────────────

def _print_chunks(chunks: list[DocumentChunk], label: str) -> None:
    """Print every chunk so the user can see exactly what will be uploaded."""
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"  Total chunks: {len(chunks)}")
    print(f"{'=' * 70}")
    for i, chunk in enumerate(chunks):
        meta = chunk.metadata
        topic = meta.get("topic", "")
        section = meta.get("section_title", "")
        heading = meta.get("heading_path", "")
        tokens = meta.get("token_count", "?")
        source = meta.get("source", chunk.source or "")

        parts = [f"Chunk {i}"]
        if section:
            parts.append(f"section={section}")
        if topic:
            parts.append(f"topic={topic}")
        if heading:
            parts.append(f"heading={heading}")
        parts.append(f"~{tokens} tok")
        parts.append(f"src={Path(source).name if source else '?'}")

        print(f"\n--- {' | '.join(parts)} ---")
        content = chunk.content
        if len(content) > 500:
            content = content[:500] + f"\n... ({len(chunk.content)} chars total)"
        print(content)
    print(f"\n{'=' * 70}\n")


def _print_result(result: UploadResult, label: str) -> None:
    """Print an UploadResult summary."""
    status = "✅" if result.success else "❌"
    print(f"  {status} {label}")
    print(f"     success={result.success}, chunks={result.total_chunks}, "
          f"ids={len(result.document_ids)}, source={result.source}")
    if result.error:
        print(f"     error={result.error}")
    if result.metadata:
        print(f"     metadata={result.metadata}")


# ── Helper: build services ───────────────────────────────────────

def _build_embeddings():
    """Build a real Cohere embeddings instance."""
    return EmbeddingsFactory.create_cohere(
        api_key=COHERE_API_KEY,
        model_name="embed-english-v3.0",
    )


def _build_inmemory_service(
    collection: str,
    use_llm: bool = False,
) -> AssetUploadService:
    """
    Build an AssetUploadService backed by in-memory Qdrant.

    Injects real Cohere embeddings. Optionally enables LLM analysis.
    """
    embeddings = _build_embeddings()

    vectordb = VectorDBFactory.create_qdrant(
        collection_name=collection,
        embedding_dimension=EMBEDDING_DIM,
        in_memory=True,
        distance_metric=DistanceMetric.COSINE,
    )

    groq_keys = _get_groq_keys() if use_llm else None

    config = AssetUploadConfig(
        default_collection=collection,
        chunk_size=1000,
        chunk_overlap=200,
        use_smart_chunker=True,
        use_llm_analysis=use_llm,
        llm_provider="groq" if use_llm else None,
        llm_model="llama-3.1-8b-instant" if use_llm else None,
        llm_api_keys=groq_keys,
    )

    return AssetUploadService(
        config=config,
        vectordb=vectordb,
        embeddings=embeddings,
    )


def _build_cloud_service(
    collection: str,
    use_llm: bool = False,
) -> AssetUploadService:
    """
    Build an AssetUploadService backed by Qdrant Cloud.

    IMPORTANT: We inject a pre-built VectorDB with prefer_grpc=False
    because VectorDBFactory.create_qdrant() doesn't expose that param
    and the default (True) hangs on cloud.
    """
    embeddings = _build_embeddings()

    cloud_config = VectorDBConfig(
        provider=VectorDBProvider.QDRANT,
        collection_name=collection,
        embedding_dimension=EMBEDDING_DIM,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        distance_metric=DistanceMetric.COSINE,
        prefer_grpc=False,  # REST — gRPC hangs on cloud
    )
    vectordb = VectorDBFactory.create(cloud_config)

    groq_keys = _get_groq_keys() if use_llm else None

    config = AssetUploadConfig(
        default_collection=collection,
        chunk_size=1000,
        chunk_overlap=200,
        use_smart_chunker=True,
        use_llm_analysis=use_llm,
        llm_provider="groq" if use_llm else None,
        llm_model="llama-3.1-8b-instant" if use_llm else None,
        llm_api_keys=groq_keys,
    )

    return AssetUploadService(
        config=config,
        vectordb=vectordb,
        embeddings=embeddings,
    )


def _cloud_coll(name: str) -> str:
    """Run-scoped cloud collection name to avoid collisions."""
    return f"upload_{name}_{RUN_ID}"


def _wait(seconds: float = 1.5):
    """Pause for Qdrant Cloud eventual consistency."""
    time.sleep(seconds)


def _safe_cloud_cleanup(service: AssetUploadService, collection: str):
    """Silently delete a cloud collection."""
    try:
        service.delete_collection(collection)
    except Exception:
        pass


# ════════════════════════════════════════════════════════════════
#  PART A — In-memory Qdrant (no LLM)
# ════════════════════════════════════════════════════════════════

def test_01_upload_txt_inmemory():
    """Test 1: Upload TXT → chunk → embed → in-memory Qdrant → verify."""
    print("\n" + "=" * 70)
    print("  TEST 1: Upload TXT file (in-memory)")
    print("=" * 70)

    coll = f"test_txt_{RUN_ID}"
    svc = _build_inmemory_service(coll)

    # Step 1 — Preview chunks
    chunks = svc._smart_chunker.chunk_file(TXT_FILE)
    _print_chunks(chunks, f"TXT Chunks Preview — {TXT_FILE.name}")
    assert len(chunks) >= 1, "Expected at least 1 chunk from TXT file"

    # Step 2 — Upload
    result = svc.upload_file(TXT_FILE, collection_name=coll)
    _print_result(result, "upload_file(TXT)")
    assert result.success, f"Upload failed: {result.error}"
    assert result.total_chunks >= 1

    # Step 3 — Retrieve with embeddings
    retrieved = svc._vectordb.get_by_ids(
        result.document_ids, collection_name=coll, include_embeddings=True
    )
    assert len(retrieved) == result.total_chunks
    for doc in retrieved:
        assert doc.embedding is not None, "Missing embedding"
        assert len(doc.embedding) == EMBEDDING_DIM, f"Wrong dim: {len(doc.embedding)}"
    print(f"  Retrieved {len(retrieved)} docs, all with {EMBEDDING_DIM}-dim embeddings ✓")

    print("  ✅ TEST 1 PASSED\n")


def test_02_upload_csv_inmemory():
    """Test 2: Upload CSV → row-per-chunk → embed → in-memory Qdrant."""
    print("\n" + "=" * 70)
    print("  TEST 2: Upload CSV file (in-memory)")
    print("=" * 70)

    coll = f"test_csv_{RUN_ID}"
    svc = _build_inmemory_service(coll)

    chunks = svc._smart_chunker.chunk_file(CSV_FILE)
    _print_chunks(chunks, f"CSV Chunks Preview — {CSV_FILE.name}")
    assert len(chunks) >= 1, "Expected at least 1 CSV chunk"

    result = svc.upload_file(CSV_FILE, collection_name=coll)
    _print_result(result, "upload_file(CSV)")
    assert result.success, f"Upload failed: {result.error}"
    assert result.total_chunks >= 1

    retrieved = svc._vectordb.get_by_ids(
        result.document_ids, collection_name=coll, include_embeddings=True
    )
    assert len(retrieved) == result.total_chunks
    for doc in retrieved:
        assert doc.embedding is not None
        assert len(doc.embedding) == EMBEDDING_DIM
    print(f"  Retrieved {len(retrieved)} docs, embeddings verified ✓")

    print("  ✅ TEST 2 PASSED\n")


def test_03_upload_json_inmemory():
    """Test 3: Upload JSON → entry-per-chunk → embed → in-memory Qdrant."""
    print("\n" + "=" * 70)
    print("  TEST 3: Upload JSON file (in-memory)")
    print("=" * 70)

    coll = f"test_json_{RUN_ID}"
    svc = _build_inmemory_service(coll)

    chunks = svc._smart_chunker.chunk_file(JSON_FILE)
    _print_chunks(chunks, f"JSON Chunks Preview — {JSON_FILE.name}")
    assert len(chunks) >= 1, "Expected at least 1 JSON chunk"

    result = svc.upload_file(JSON_FILE, collection_name=coll)
    _print_result(result, "upload_file(JSON)")
    assert result.success, f"Upload failed: {result.error}"

    retrieved = svc._vectordb.get_by_ids(
        result.document_ids, collection_name=coll, include_embeddings=True
    )
    assert len(retrieved) == result.total_chunks
    for doc in retrieved:
        assert doc.embedding is not None
        assert len(doc.embedding) == EMBEDDING_DIM
    print(f"  Retrieved {len(retrieved)} docs, embeddings verified ✓")

    print("  ✅ TEST 3 PASSED\n")


def test_04_upload_raw_text_inmemory():
    """Test 4: Upload raw text string → chunk → embed → in-memory Qdrant."""
    print("\n" + "=" * 70)
    print("  TEST 4: Upload raw text (in-memory)")
    print("=" * 70)

    coll = f"test_rawtext_{RUN_ID}"
    svc = _build_inmemory_service(coll)

    text = (
        "Garvit Batra is a software developer specializing in AI and machine learning. "
        "He has experience building RAG pipelines, LLM integrations, and full-stack applications. "
        "His tech stack includes Python, FastAPI, React, and various cloud services. "
        "He is passionate about creating intelligent systems that solve real-world problems."
    )

    result = svc.upload_text(
        text,
        collection_name=coll,
        metadata={"type": "bio", "author": "manual"},
        source="manual_input",
    )
    _print_result(result, "upload_text(raw)")
    assert result.success, f"Upload failed: {result.error}"
    assert result.total_chunks >= 1

    retrieved = svc._vectordb.get_by_ids(
        result.document_ids, collection_name=coll, include_embeddings=True
    )
    assert len(retrieved) == result.total_chunks
    for doc in retrieved:
        assert doc.embedding is not None
        assert len(doc.embedding) == EMBEDDING_DIM
        print(f"    chunk: {doc.content[:80]}...")
    print(f"  Retrieved {len(retrieved)} docs, embeddings verified ✓")

    print("  ✅ TEST 4 PASSED\n")


def test_05_upload_directory_inmemory():
    """Test 5: Upload entire test data directory → chunk all files → embed → Qdrant."""
    print("\n" + "=" * 70)
    print("  TEST 5: Upload directory (in-memory)")
    print("=" * 70)

    coll = f"test_dir_{RUN_ID}"
    svc = _build_inmemory_service(coll)

    results = svc.upload_directory(
        DATA_DIR,
        collection_name=coll,
        recursive=False,
        extensions=["txt", "csv", "json"],
    )
    assert len(results) >= 1, "Expected at least 1 file result"

    total_chunks = 0
    for r in results:
        _print_result(r, f"  dir upload → {r.source}")
        assert r.success, f"Failed: {r.error}"
        total_chunks += r.total_chunks

    print(f"\n  Total files processed: {len(results)}")
    print(f"  Total chunks uploaded: {total_chunks}")

    # Verify count
    count = svc._vectordb.count(collection_name=coll)
    assert count == total_chunks, f"Count mismatch: {count} != {total_chunks}"
    print(f"  VectorDB count: {count} ✓")

    print("  ✅ TEST 5 PASSED\n")


def test_06_search_after_upload_inmemory():
    """Test 6: Upload TXT, then do similarity search on the uploaded data."""
    print("\n" + "=" * 70)
    print("  TEST 6: Search after upload (in-memory)")
    print("=" * 70)

    coll = f"test_search_{RUN_ID}"
    svc = _build_inmemory_service(coll)

    # Upload the TXT file
    result = svc.upload_file(TXT_FILE, collection_name=coll)
    assert result.success, f"Upload failed: {result.error}"
    print(f"  Uploaded {result.total_chunks} chunks from {TXT_FILE.name}")

    # Embed a query
    query = "What are the skills and experience?"
    query_emb = svc._embeddings.embed_query(query, input_type=EmbeddingInputType.SEARCH_QUERY)
    assert len(query_emb) == EMBEDDING_DIM

    # Search
    results = svc._vectordb.search(
        query_embedding=query_emb,
        k=3,
        collection_name=coll,
    )
    assert len(results) >= 1, "Expected at least 1 search result"

    print(f"\n  Query: '{query}'")
    print(f"  Top-{len(results)} results:")
    for r in results:
        print(f"    score={r.score:.4f}  content={r.content[:100]}...")
    print(f"\n  Search returned {len(results)} results ✓")

    print("  ✅ TEST 6 PASSED\n")


# ════════════════════════════════════════════════════════════════
#  PART B — LLM-assisted chunking (in-memory Qdrant)
# ════════════════════════════════════════════════════════════════

def test_07_llm_topical_txt_inmemory():
    """Test 7: TXT + LLM topical analysis → chunk → embed → in-memory."""
    print("\n" + "=" * 70)
    print("  TEST 7: TXT + LLM topical chunking (in-memory)")
    print("=" * 70)

    coll = f"test_llm_txt_{RUN_ID}"
    svc = _build_inmemory_service(coll, use_llm=True)

    # Preview chunks with LLM
    chunks = svc._smart_chunker.chunk_file(TXT_FILE)
    _print_chunks(chunks, f"LLM Topical Chunks — {TXT_FILE.name}")
    assert len(chunks) >= 1

    # Check for topic labels from LLM
    topics = [c.metadata.get("topic") for c in chunks if c.metadata.get("topic")]
    print(f"  Topics detected by LLM: {topics}")

    # Upload
    result = svc.upload_file(TXT_FILE, collection_name=coll)
    _print_result(result, "upload_file(TXT + LLM)")
    assert result.success

    # Verify embeddings
    retrieved = svc._vectordb.get_by_ids(
        result.document_ids, collection_name=coll, include_embeddings=True
    )
    for doc in retrieved:
        assert doc.embedding is not None
        assert len(doc.embedding) == EMBEDDING_DIM
    print(f"  All {len(retrieved)} chunks have {EMBEDDING_DIM}-dim embeddings ✓")

    print("  ✅ TEST 7 PASSED\n")


def test_08_llm_structural_pdf_inmemory():
    """Test 8: PDF + LLM structural analysis → chunk → embed → in-memory."""
    print("\n" + "=" * 70)
    print("  TEST 8: PDF + LLM structural chunking (in-memory)")
    print("=" * 70)

    coll = f"test_llm_pdf_{RUN_ID}"
    svc = _build_inmemory_service(coll, use_llm=True)

    # Preview chunks with LLM structural analysis
    chunks = svc._smart_chunker.chunk_file(PDF_FILE)
    _print_chunks(chunks, f"LLM Structural Chunks — {PDF_FILE.name}")
    assert len(chunks) >= 1

    # Upload
    result = svc.upload_file(PDF_FILE, collection_name=coll)
    _print_result(result, "upload_file(PDF + LLM)")
    assert result.success

    # Verify
    retrieved = svc._vectordb.get_by_ids(
        result.document_ids, collection_name=coll, include_embeddings=True
    )
    for doc in retrieved:
        assert doc.embedding is not None
        assert len(doc.embedding) == EMBEDDING_DIM
    print(f"  All {len(retrieved)} chunks have {EMBEDDING_DIM}-dim embeddings ✓")

    print("  ✅ TEST 8 PASSED\n")


def test_09_llm_vs_nollm_comparison():
    """Test 9: Compare LLM-assisted vs rule-based chunking side by side."""
    print("\n" + "=" * 70)
    print("  TEST 9: LLM vs No-LLM chunk comparison")
    print("=" * 70)

    # Rule-based
    svc_nollm = _build_inmemory_service(f"cmp_nollm_{RUN_ID}", use_llm=False)
    chunks_nollm = svc_nollm._smart_chunker.chunk_file(TXT_FILE)

    # LLM-assisted
    svc_llm = _build_inmemory_service(f"cmp_llm_{RUN_ID}", use_llm=True)
    chunks_llm = svc_llm._smart_chunker.chunk_file(TXT_FILE)

    _print_chunks(chunks_nollm, f"Rule-based chunks — {TXT_FILE.name}")
    _print_chunks(chunks_llm, f"LLM-assisted chunks — {TXT_FILE.name}")

    print(f"  Comparison:")
    print(f"    Rule-based : {len(chunks_nollm)} chunks")
    print(f"    LLM-assisted: {len(chunks_llm)} chunks")

    # Both should produce at least 1 chunk
    assert len(chunks_nollm) >= 1
    assert len(chunks_llm) >= 1

    # LLM chunks may have topic labels
    llm_topics = [c.metadata.get("topic") for c in chunks_llm if c.metadata.get("topic")]
    nollm_topics = [c.metadata.get("topic") for c in chunks_nollm if c.metadata.get("topic")]
    print(f"    LLM topics : {llm_topics}")
    print(f"    Rule topics: {nollm_topics}")

    print("  ✅ TEST 9 PASSED\n")


# ════════════════════════════════════════════════════════════════
#  PART C — Qdrant Cloud (no LLM)
# ════════════════════════════════════════════════════════════════

def test_10_upload_txt_cloud():
    """Test 10: Upload TXT → chunk → embed → Qdrant Cloud → verify."""
    print("\n" + "=" * 70)
    print("  TEST 10: Upload TXT file (Qdrant Cloud)")
    print("=" * 70)

    coll = _cloud_coll("txt")
    svc = _build_cloud_service(coll)
    try:
        # Preview chunks
        chunks = svc._smart_chunker.chunk_file(TXT_FILE)
        _print_chunks(chunks, f"TXT Chunks Preview — {TXT_FILE.name}")

        # Upload
        result = svc.upload_file(TXT_FILE, collection_name=coll)
        _print_result(result, "upload_file(TXT → Cloud)")
        assert result.success, f"Upload failed: {result.error}"
        assert result.total_chunks >= 1
        _wait()

        # Retrieve with embeddings from cloud
        retrieved = svc._vectordb.get_by_ids(
            result.document_ids, collection_name=coll, include_embeddings=True
        )
        assert len(retrieved) == result.total_chunks
        for doc in retrieved:
            assert doc.embedding is not None
            assert len(doc.embedding) == EMBEDDING_DIM
        print(f"  Cloud: Retrieved {len(retrieved)} docs, {EMBEDDING_DIM}-dim embeddings ✓")

        # Cloud count
        count = svc._vectordb.count(collection_name=coll)
        assert count == result.total_chunks
        print(f"  Cloud count: {count} ✓")

        print("  ✅ TEST 10 PASSED\n")
    finally:
        _safe_cloud_cleanup(svc, coll)


def test_11_upload_csv_cloud():
    """Test 11: Upload CSV → Qdrant Cloud."""
    print("\n" + "=" * 70)
    print("  TEST 11: Upload CSV file (Qdrant Cloud)")
    print("=" * 70)

    coll = _cloud_coll("csv")
    svc = _build_cloud_service(coll)
    try:
        chunks = svc._smart_chunker.chunk_file(CSV_FILE)
        _print_chunks(chunks, f"CSV Chunks Preview — {CSV_FILE.name}")

        result = svc.upload_file(CSV_FILE, collection_name=coll)
        _print_result(result, "upload_file(CSV → Cloud)")
        assert result.success, f"Upload failed: {result.error}"
        _wait()

        retrieved = svc._vectordb.get_by_ids(
            result.document_ids, collection_name=coll, include_embeddings=True
        )
        assert len(retrieved) == result.total_chunks
        for doc in retrieved:
            assert doc.embedding is not None
            assert len(doc.embedding) == EMBEDDING_DIM
        print(f"  Cloud: Retrieved {len(retrieved)} docs, embeddings verified ✓")

        print("  ✅ TEST 11 PASSED\n")
    finally:
        _safe_cloud_cleanup(svc, coll)


def test_12_upload_json_cloud():
    """Test 12: Upload JSON → Qdrant Cloud."""
    print("\n" + "=" * 70)
    print("  TEST 12: Upload JSON file (Qdrant Cloud)")
    print("=" * 70)

    coll = _cloud_coll("json")
    svc = _build_cloud_service(coll)
    try:
        chunks = svc._smart_chunker.chunk_file(JSON_FILE)
        _print_chunks(chunks, f"JSON Chunks Preview — {JSON_FILE.name}")

        result = svc.upload_file(JSON_FILE, collection_name=coll)
        _print_result(result, "upload_file(JSON → Cloud)")
        assert result.success, f"Upload failed: {result.error}"
        _wait()

        retrieved = svc._vectordb.get_by_ids(
            result.document_ids, collection_name=coll, include_embeddings=True
        )
        assert len(retrieved) == result.total_chunks
        for doc in retrieved:
            assert doc.embedding is not None
            assert len(doc.embedding) == EMBEDDING_DIM
        print(f"  Cloud: Retrieved {len(retrieved)} docs, embeddings verified ✓")

        print("  ✅ TEST 12 PASSED\n")
    finally:
        _safe_cloud_cleanup(svc, coll)


def test_13_search_after_upload_cloud():
    """Test 13: Upload TXT to Cloud, then similarity search."""
    print("\n" + "=" * 70)
    print("  TEST 13: Search after upload (Qdrant Cloud)")
    print("=" * 70)

    coll = _cloud_coll("search")
    svc = _build_cloud_service(coll)
    try:
        result = svc.upload_file(TXT_FILE, collection_name=coll)
        assert result.success, f"Upload failed: {result.error}"
        print(f"  Uploaded {result.total_chunks} chunks to cloud")
        _wait(2.0)  # extra wait for indexing

        query = "What programming languages does the person know?"
        query_emb = svc._embeddings.embed_query(
            query, input_type=EmbeddingInputType.SEARCH_QUERY
        )

        results = svc._vectordb.search(
            query_embedding=query_emb,
            k=3,
            collection_name=coll,
        )
        assert len(results) >= 1, "Expected at least 1 search result"

        print(f"\n  Query: '{query}'")
        print(f"  Top-{len(results)} results from cloud:")
        for r in results:
            print(f"    score={r.score:.4f}  content={r.content[:100]}...")

        print("  ✅ TEST 13 PASSED\n")
    finally:
        _safe_cloud_cleanup(svc, coll)


# ════════════════════════════════════════════════════════════════
#  PART D — LLM + Cloud combined
# ════════════════════════════════════════════════════════════════

def test_14_llm_plus_cloud():
    """Test 14: LLM-assisted chunking + Cloud upload + Search."""
    print("\n" + "=" * 70)
    print("  TEST 14: LLM chunking + Qdrant Cloud upload + Search")
    print("=" * 70)

    coll = _cloud_coll("llm_cloud")
    svc = _build_cloud_service(coll, use_llm=True)
    try:
        # Preview LLM chunks
        chunks = svc._smart_chunker.chunk_file(TXT_FILE)
        _print_chunks(chunks, f"LLM Chunks for Cloud — {TXT_FILE.name}")

        # Upload
        result = svc.upload_file(TXT_FILE, collection_name=coll)
        _print_result(result, "upload_file(TXT + LLM → Cloud)")
        assert result.success, f"Upload failed: {result.error}"
        _wait(2.0)

        # Verify embeddings from cloud
        retrieved = svc._vectordb.get_by_ids(
            result.document_ids, collection_name=coll, include_embeddings=True
        )
        for doc in retrieved:
            assert doc.embedding is not None
            assert len(doc.embedding) == EMBEDDING_DIM

        # Search on cloud data
        query = "education and background"
        query_emb = svc._embeddings.embed_query(
            query, input_type=EmbeddingInputType.SEARCH_QUERY
        )
        results = svc._vectordb.search(
            query_embedding=query_emb,
            k=3,
            collection_name=coll,
        )
        assert len(results) >= 1

        print(f"\n  Query: '{query}'")
        print(f"  Top-{len(results)} cloud results:")
        for r in results:
            print(f"    score={r.score:.4f}  content={r.content[:100]}...")

        print("  ✅ TEST 14 PASSED\n")
    finally:
        _safe_cloud_cleanup(svc, coll)


# ════════════════════════════════════════════════════════════════
#  Main runner
# ════════════════════════════════════════════════════════════════

def main():
    # ── Pre-flight checks ──────────────────────────────────────
    if not COHERE_API_KEY:
        print("❌  COHERE_API_KEY not set. Cannot run any tests.")
        sys.exit(1)

    groq_keys = _get_groq_keys()
    has_llm = len(groq_keys) > 0
    has_cloud = bool(QDRANT_URL and QDRANT_API_KEY)

    print(f"\n{'=' * 70}")
    print("  Asset Upload Service — E2E Test Suite")
    print(f"{'=' * 70}")
    print(f"  Run ID         : {RUN_ID}")
    print(f"  Cohere API     : ✅ set")
    print(f"  Groq LLM keys  : {'✅ ' + str(len(groq_keys)) + ' key(s)' if has_llm else '⚠️  not set (Part B/D skipped)'}")
    print(f"  Qdrant Cloud   : {'✅ ' + QDRANT_URL if has_cloud else '⚠️  not set (Part C/D skipped)'}")
    print(f"  Test data dir  : {DATA_DIR}")
    print(f"{'=' * 70}\n")

    # Verify test data exists
    for f in [TXT_FILE, CSV_FILE, JSON_FILE, PDF_FILE]:
        assert f.exists(), f"Test data missing: {f}"

    passed = 0
    failed = 0
    skipped = 0

    def _run(test_fn, requires_llm=False, requires_cloud=False):
        nonlocal passed, failed, skipped
        if requires_llm and not has_llm:
            print(f"  ⏭️  SKIPPED {test_fn.__name__} (no Groq keys)\n")
            skipped += 1
            return
        if requires_cloud and not has_cloud:
            print(f"  ⏭️  SKIPPED {test_fn.__name__} (no Qdrant Cloud)\n")
            skipped += 1
            return
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            import traceback
            print(f"  ❌ FAILED: {test_fn.__name__}")
            traceback.print_exc()
            print()

    # Part A — In-memory
    print("\n" + "─" * 70)
    print("  PART A: In-memory Qdrant (no LLM)")
    print("─" * 70)
    _run(test_01_upload_txt_inmemory)
    _run(test_02_upload_csv_inmemory)
    _run(test_03_upload_json_inmemory)
    _run(test_04_upload_raw_text_inmemory)
    _run(test_05_upload_directory_inmemory)
    _run(test_06_search_after_upload_inmemory)

    # Part B — LLM chunking (in-memory)
    print("\n" + "─" * 70)
    print("  PART B: LLM-assisted chunking (in-memory Qdrant)")
    print("─" * 70)
    _run(test_07_llm_topical_txt_inmemory, requires_llm=True)
    _run(test_08_llm_structural_pdf_inmemory, requires_llm=True)
    _run(test_09_llm_vs_nollm_comparison, requires_llm=True)

    # Part C — Cloud
    print("\n" + "─" * 70)
    print("  PART C: Qdrant Cloud (no LLM)")
    print("─" * 70)
    _run(test_10_upload_txt_cloud, requires_cloud=True)
    _run(test_11_upload_csv_cloud, requires_cloud=True)
    _run(test_12_upload_json_cloud, requires_cloud=True)
    _run(test_13_search_after_upload_cloud, requires_cloud=True)

    # Part D — LLM + Cloud
    print("\n" + "─" * 70)
    print("  PART D: LLM + Qdrant Cloud")
    print("─" * 70)
    _run(test_14_llm_plus_cloud, requires_llm=True, requires_cloud=True)

    # Summary
    total = passed + failed + skipped
    print("\n" + "=" * 70)
    print(f"  RESULTS: {passed} passed, {failed} failed, {skipped} skipped (of {total})")
    print("=" * 70)

    if failed == 0:
        print(f"  ✅ All {passed} tests passed!" +
              (f" ({skipped} skipped)" if skipped else ""))
    else:
        print(f"  ❌ {failed} test(s) failed")

    print("=" * 70 + "\n")

    if failed:
        sys.exit(1)


# ════════════════════════════════════════════════════════════════
#  STANDALONE: Upload all formats to Cloud — NO CLEANUP
#  Run:  PYTHONPATH=. python tests/test_asset_upload_e2e.py --persist
# ════════════════════════════════════════════════════════════════

def test_persist_all_formats_cloud():
    """
    Upload TXT, CSV, JSON, and PDF to a SINGLE Qdrant Cloud collection.
    Collections are NOT deleted — inspect them on the Qdrant dashboard.
    """
    if not COHERE_API_KEY:
        print("❌  COHERE_API_KEY not set.")
        sys.exit(1)
    if not QDRANT_URL or not QDRANT_API_KEY:
        print("❌  QDRANT_URL / QDRANT_API_KEY not set.")
        sys.exit(1)

    coll = f"persist_all_{RUN_ID}"

    print(f"\n{'=' * 70}")
    print(f"  PERSIST TEST: Upload ALL formats to Qdrant Cloud")
    print(f"  Collection : {coll}")
    print(f"  Run ID     : {RUN_ID}")
    print(f"  Qdrant URL : {QDRANT_URL}")
    print(f"  ⚠️  Collection will NOT be deleted — check your dashboard!")
    print(f"{'=' * 70}\n")

    svc = _build_cloud_service(coll, use_llm=False)

    files = [
        ("TXT", TXT_FILE),
        ("CSV", CSV_FILE),
        ("JSON", JSON_FILE),
        ("PDF", PDF_FILE),
    ]

    total_chunks = 0
    all_ids = []

    for label, fpath in files:
        print(f"\n{'─' * 70}")
        print(f"  Uploading {label}: {fpath.name}")
        print(f"{'─' * 70}")

        # Preview chunks
        chunks = svc._smart_chunker.chunk_file(fpath)
        _print_chunks(chunks, f"{label} Chunks — {fpath.name}")

        # Upload
        result = svc.upload_file(fpath, collection_name=coll)
        _print_result(result, f"upload {label} → Cloud")
        assert result.success, f"Upload failed for {label}: {result.error}"

        total_chunks += result.total_chunks
        all_ids.extend(result.document_ids)
        _wait()

    # Also upload a raw text snippet
    print(f"\n{'─' * 70}")
    print(f"  Uploading raw text")
    print(f"{'─' * 70}")
    raw_result = svc.upload_text(
        "Garvit Batra is a software developer at Microsoft, specializing in "
        "AI/ML, RAG pipelines, and full-stack development with Python and React.",
        collection_name=coll,
        metadata={"type": "bio", "format": "raw_text"},
        source="manual_input",
    )
    _print_result(raw_result, "upload raw text → Cloud")
    assert raw_result.success
    total_chunks += raw_result.total_chunks
    all_ids.extend(raw_result.document_ids)
    _wait()

    # Verify total count on cloud
    count = svc._vectordb.count(collection_name=coll)
    print(f"\n  Cloud collection '{coll}' total vectors: {count}")
    assert count == total_chunks, f"Count mismatch: {count} != {total_chunks}"

    # Quick search to confirm it all works
    query = "What projects has Garvit built?"
    query_emb = svc._embeddings.embed_query(query, input_type=EmbeddingInputType.SEARCH_QUERY)
    results = svc._vectordb.search(query_embedding=query_emb, k=5, collection_name=coll)

    print(f"\n  🔍 Sample search: '{query}'")
    for r in results:
        src = r.metadata.get("source", "?")
        print(f"    score={r.score:.4f}  src={Path(src).name}  content={r.content[:80]}...")

    print(f"\n{'=' * 70}")
    print(f"  ✅ PERSIST TEST PASSED")
    print(f"  📌 Collection '{coll}' preserved on Qdrant Cloud")
    print(f"     Total vectors : {count}")
    print(f"     Total IDs     : {len(all_ids)}")
    print(f"     Formats       : TXT, CSV, JSON, PDF, raw text")
    print(f"     → Go to your Qdrant dashboard to inspect the data!")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    if "--persist" in sys.argv:
        test_persist_all_formats_cloud()
    else:
        main()
