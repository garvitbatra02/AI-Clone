"""
End-to-end tests for the Asset Upload Dashboard API.

Tests every endpoint of the AssetUploadService dashboard through the
FastAPI TestClient, using an in-memory Qdrant backend with real Cohere
embeddings — exactly the same stack the core AssetUploadService tests use.

Covers:
  Part A — Collection Management
    1.  GET  /api/assets/collections             (empty initially)
    2.  POST /api/assets/collections             (create collection)
    3.  POST /api/assets/collections             (duplicate → already_exists)
    4.  GET  /api/assets/collections/{name}      (single collection stats)
    5.  GET  /api/assets/collections/{name}      (non-existent → 404)
    6.  DELETE /api/assets/collections/{name}     (delete collection)
    7.  DELETE /api/assets/collections/{name}     (delete again → 404)

  Part B — Upload Previews
    8.  GET  /api/assets/uploads/supported-types  (list extensions)
    9.  POST /api/assets/uploads/preview/local    (preview TXT)
    10. POST /api/assets/uploads/preview/local    (preview CSV)
    11. POST /api/assets/uploads/preview/local    (preview JSON)
    12. POST /api/assets/uploads/preview/local    (non-existent file → 404)
    13. POST /api/assets/uploads/preview          (multipart TXT preview)
    14. POST /api/assets/uploads/preview          (multipart CSV preview)
    15. POST /api/assets/uploads/preview          (unsupported ext → 400)

  Part C — File Uploads (local path)
    16. POST /api/assets/uploads/file/local       (upload TXT)
    17. POST /api/assets/uploads/file/local       (upload CSV)
    18. POST /api/assets/uploads/file/local       (upload JSON)
    19. POST /api/assets/uploads/file/local       (non-existent → 404)
    20. GET  /api/assets/collections/{name}       (verify vector count after upload)

  Part D — File Uploads (multipart)
    21. POST /api/assets/uploads/file             (multipart TXT upload)
    22. POST /api/assets/uploads/file             (multipart TXT + collection_name)

  Part E — Text Uploads
    23. POST /api/assets/uploads/text             (single text)
    24. POST /api/assets/uploads/text             (with metadata + source)
    25. POST /api/assets/uploads/texts            (batch texts)

  Part F — Directory Upload
    26. POST /api/assets/uploads/directory/local   (upload directory)
    27. POST /api/assets/uploads/directory/local   (with extension filter)
    28. POST /api/assets/uploads/directory/local   (non-existent dir → 404)

  Part G — Full Dashboard Workflow
    29. Create collection → preview file → upload file → verify stats
    30. Upload text → verify stats → delete collection → verify deleted

Requires (.env):
    COHERE_API_KEY  — Cohere embed-english-v3.0

Run:
    PYTHONPATH=. python tests/test_dashboard_e2e.py
    PYTHONPATH=. python -m pytest tests/test_dashboard_e2e.py -v --tb=short
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Imports ──────────────────────────────────────────────────────

from fastapi import FastAPI
from fastapi.testclient import TestClient

from RAGService.Data.VectorDB import (
    VectorDBFactory,
    VectorDBProvider,
    DistanceMetric,
)
from RAGService.Data.Embeddings import (
    EmbeddingsFactory,
    EmbeddingProvider,
)
from RAGService.Data.services.asset_upload_service import (
    AssetUploadConfig,
    AssetUploadService,
)
from AssetUploadService.services.dashboard_service import DashboardService
from AssetUploadService.Server.routes.collections import router as collections_router
from AssetUploadService.Server.routes.uploads import router as uploads_router

# ── Configuration ────────────────────────────────────────────────

EMBEDDING_DIM = 1024  # Cohere embed-english-v3.0
RUN_ID = uuid.uuid4().hex[:8]

# Test data directory
DATA_DIR = Path(__file__).parent / "document_loaders_tests"
TXT_FILE = DATA_DIR / "my all details.txt"
CSV_FILE = DATA_DIR / "Personal_TrainingData.csv"
JSON_FILE = DATA_DIR / "conversation.json"

# ── Helpers ──────────────────────────────────────────────────────

_test_service: DashboardService | None = None


def _build_test_service() -> DashboardService:
    """Build a DashboardService backed by in-memory Qdrant + real Cohere embeddings."""
    global _test_service
    if _test_service is not None:
        return _test_service

    embeddings = EmbeddingsFactory.create_cohere(model_name="embed-english-v3.0")

    vectordb = VectorDBFactory.create_from_env(
        provider=VectorDBProvider.QDRANT,
        collection_name="test_default",
        embedding_dimension=EMBEDDING_DIM,
        in_memory=True,
        distance_metric=DistanceMetric.COSINE,
    )

    config = AssetUploadConfig(
        default_collection="test_default",
        chunk_size=1000,
        chunk_overlap=200,
        use_smart_chunker=True,
        use_llm_analysis=False,
    )

    core = AssetUploadService(config=config, vectordb=vectordb, embeddings=embeddings)
    _test_service = DashboardService(core_service=core)
    return _test_service


def _build_test_app() -> FastAPI:
    """
    Build a self-contained FastAPI app with the asset routers.
    
    Monkey-patches the route modules' _get_dashboard_service() to
    return our in-memory test service instead of the global singleton.
    """
    service = _build_test_service()

    # Patch the lazy getters in the route modules
    import AssetUploadService.Server.routes.collections as coll_mod
    import AssetUploadService.Server.routes.uploads as upl_mod
    coll_mod._get_dashboard_service = lambda: service
    upl_mod._get_dashboard_service = lambda: service

    app = FastAPI()
    app.include_router(collections_router, prefix="/api")
    app.include_router(uploads_router, prefix="/api")
    return app


# Build once for all tests
_app = _build_test_app()
client = TestClient(_app)


def _coll(name: str) -> str:
    """Run-scoped collection name to avoid collisions."""
    return f"test_{name}_{RUN_ID}"


def _print(msg: str):
    print(f"  {msg}")


# ════════════════════════════════════════════════════════════════
#  PART A — Collection Management
# ════════════════════════════════════════════════════════════════


def test_01_list_collections_empty():
    """Test 1: GET /api/assets/collections — initially empty."""
    print("\n" + "=" * 70)
    print("  TEST 1: List collections (empty)")
    print("=" * 70)

    r = client.get("/api/assets/collections")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()
    assert "collections" in data
    assert "total" in data
    _print(f"Collections: {data['total']} — {data['collections']}")
    _print("✅ TEST 1 PASSED")


def test_02_create_collection():
    """Test 2: POST /api/assets/collections — create new."""
    print("\n" + "=" * 70)
    print("  TEST 2: Create collection")
    print("=" * 70)

    name = _coll("created")
    r = client.post("/api/assets/collections", json={"name": name})
    assert r.status_code == 201, f"Expected 201, got {r.status_code}: {r.text}"
    data = r.json()
    assert data["created"] is True
    assert data["collection"] == name
    assert data.get("reason") is None
    _print(f"Created: {data}")
    _print("✅ TEST 2 PASSED")


def test_03_create_collection_duplicate():
    """Test 3: POST /api/assets/collections — duplicate → already_exists."""
    print("\n" + "=" * 70)
    print("  TEST 3: Create duplicate collection")
    print("=" * 70)

    name = _coll("created")  # same as test 2
    r = client.post("/api/assets/collections", json={"name": name})
    assert r.status_code == 201, f"Expected 201, got {r.status_code}: {r.text}"
    data = r.json()
    assert data["created"] is False
    assert data["reason"] == "already_exists"
    _print(f"Duplicate response: {data}")
    _print("✅ TEST 3 PASSED")


def test_04_get_collection_stats():
    """Test 4: GET /api/assets/collections/{name} — existing collection."""
    print("\n" + "=" * 70)
    print("  TEST 4: Get collection stats")
    print("=" * 70)

    name = _coll("created")
    r = client.get(f"/api/assets/collections/{name}")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()
    assert data["collection"] == name
    assert data["exists"] is True
    assert "vector_count" in data
    assert "dimension" in data
    assert "distance_metric" in data
    _print(f"Stats: {data}")
    _print("✅ TEST 4 PASSED")


def test_05_get_collection_not_found():
    """Test 5: GET /api/assets/collections/{name} — non-existent → 404."""
    print("\n" + "=" * 70)
    print("  TEST 5: Get non-existent collection (404)")
    print("=" * 70)

    r = client.get("/api/assets/collections/this_does_not_exist_xyz")
    assert r.status_code == 404, f"Expected 404, got {r.status_code}: {r.text}"
    _print(f"404 response: {r.json()}")
    _print("✅ TEST 5 PASSED")


def test_06_delete_collection():
    """Test 6: DELETE /api/assets/collections/{name} — success."""
    print("\n" + "=" * 70)
    print("  TEST 6: Delete collection")
    print("=" * 70)

    name = _coll("to_delete")
    # Create it first
    client.post("/api/assets/collections", json={"name": name})

    r = client.delete(f"/api/assets/collections/{name}")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()
    assert data["deleted"] is True
    assert data["collection"] == name
    _print(f"Deleted: {data}")

    # Verify it's gone
    r2 = client.get(f"/api/assets/collections/{name}")
    assert r2.status_code == 404
    _print("Confirmed gone via GET → 404")
    _print("✅ TEST 6 PASSED")


def test_07_delete_collection_not_found():
    """Test 7: DELETE /api/assets/collections/{name} — non-existent.
    
    Note: Qdrant's delete is idempotent — deleting a non-existent
    collection returns True (no-op).  The route returns 200 in this case.
    We verify the collection still doesn't appear in the list.
    """
    print("\n" + "=" * 70)
    print("  TEST 7: Delete non-existent collection (idempotent)")
    print("=" * 70)

    r = client.delete("/api/assets/collections/nonexistent_xyz")
    # Qdrant treats delete as idempotent — returns success even for non-existent
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    _print(f"Response: {r.json()}")

    # Verify it doesn't appear in the collection list
    r2 = client.get("/api/assets/collections")
    names = [c["collection"] for c in r2.json()["collections"]]
    assert "nonexistent_xyz" not in names
    _print("Confirmed not in collection list")
    _print("✅ TEST 7 PASSED")


# ════════════════════════════════════════════════════════════════
#  PART B — Upload Previews
# ════════════════════════════════════════════════════════════════


def test_08_supported_types():
    """Test 8: GET /api/assets/uploads/supported-types."""
    print("\n" + "=" * 70)
    print("  TEST 8: Supported file types")
    print("=" * 70)

    r = client.get("/api/assets/uploads/supported-types")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()
    exts = data["extensions"]
    assert isinstance(exts, list)
    assert len(exts) >= 5  # at least txt, md, json, csv, pdf
    assert "txt" in exts
    assert "pdf" in exts
    assert "csv" in exts
    assert "json" in exts
    _print(f"Supported extensions: {exts}")
    _print("✅ TEST 8 PASSED")


def test_09_preview_local_txt():
    """Test 9: POST /api/assets/uploads/preview/local — TXT file."""
    print("\n" + "=" * 70)
    print("  TEST 9: Preview local TXT file")
    print("=" * 70)

    r = client.post("/api/assets/uploads/preview/local", json={
        "file_path": str(TXT_FILE),
    })
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()
    assert data["file_name"] == TXT_FILE.name
    assert data["file_type"] == "txt"
    assert data["file_size_bytes"] > 0
    assert data["total_chunks"] >= 1
    assert data["estimated_tokens"] > 0
    assert len(data["chunk_previews"]) >= 1
    _print(f"Preview: {data['file_name']}, type={data['file_type']}, "
           f"size={data['file_size_bytes']}B, chunks={data['total_chunks']}, "
           f"~tokens={data['estimated_tokens']}")
    for cp in data["chunk_previews"][:2]:
        _print(f"  Chunk {cp['index']}: {cp['char_count']} chars — "
               f"{cp['content_preview'][:60]}...")
    _print("✅ TEST 9 PASSED")


def test_10_preview_local_csv():
    """Test 10: POST /api/assets/uploads/preview/local — CSV file."""
    print("\n" + "=" * 70)
    print("  TEST 10: Preview local CSV file")
    print("=" * 70)

    r = client.post("/api/assets/uploads/preview/local", json={
        "file_path": str(CSV_FILE),
    })
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()
    assert data["file_type"] == "csv"
    assert data["total_chunks"] >= 1
    _print(f"Preview: {data['file_name']}, chunks={data['total_chunks']}, "
           f"~tokens={data['estimated_tokens']}")
    _print("✅ TEST 10 PASSED")


def test_11_preview_local_json():
    """Test 11: POST /api/assets/uploads/preview/local — JSON file."""
    print("\n" + "=" * 70)
    print("  TEST 11: Preview local JSON file")
    print("=" * 70)

    r = client.post("/api/assets/uploads/preview/local", json={
        "file_path": str(JSON_FILE),
    })
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()
    assert data["file_type"] == "json"
    assert data["total_chunks"] >= 1
    _print(f"Preview: {data['file_name']}, chunks={data['total_chunks']}, "
           f"~tokens={data['estimated_tokens']}")
    _print("✅ TEST 11 PASSED")


def test_12_preview_local_not_found():
    """Test 12: POST /api/assets/uploads/preview/local — non-existent file → 404."""
    print("\n" + "=" * 70)
    print("  TEST 12: Preview non-existent file (404)")
    print("=" * 70)

    r = client.post("/api/assets/uploads/preview/local", json={
        "file_path": "/tmp/this_does_not_exist_xyz.txt",
    })
    assert r.status_code == 404, f"Expected 404, got {r.status_code}: {r.text}"
    _print(f"404 response: {r.json()}")
    _print("✅ TEST 12 PASSED")


def test_13_preview_multipart_txt():
    """Test 13: POST /api/assets/uploads/preview — multipart TXT preview."""
    print("\n" + "=" * 70)
    print("  TEST 13: Preview multipart TXT upload")
    print("=" * 70)

    with open(TXT_FILE, "rb") as f:
        r = client.post(
            "/api/assets/uploads/preview",
            files={"file": (TXT_FILE.name, f, "text/plain")},
        )
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()
    assert data["file_name"] == TXT_FILE.name
    assert data["file_type"] == "txt"
    assert data["total_chunks"] >= 1
    assert data["estimated_tokens"] > 0
    _print(f"Multipart preview: chunks={data['total_chunks']}, "
           f"~tokens={data['estimated_tokens']}")
    _print("✅ TEST 13 PASSED")


def test_14_preview_multipart_csv():
    """Test 14: POST /api/assets/uploads/preview — multipart CSV preview."""
    print("\n" + "=" * 70)
    print("  TEST 14: Preview multipart CSV upload")
    print("=" * 70)

    with open(CSV_FILE, "rb") as f:
        r = client.post(
            "/api/assets/uploads/preview",
            files={"file": (CSV_FILE.name, f, "text/csv")},
        )
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()
    assert data["file_type"] == "csv"
    assert data["total_chunks"] >= 1
    _print(f"CSV multipart preview: chunks={data['total_chunks']}")
    _print("✅ TEST 14 PASSED")


def test_15_preview_multipart_unsupported():
    """Test 15: POST /api/assets/uploads/preview — unsupported extension → 400/500."""
    print("\n" + "=" * 70)
    print("  TEST 15: Preview unsupported file type")
    print("=" * 70)

    content = b"Some binary content that is not a real file"
    r = client.post(
        "/api/assets/uploads/preview",
        files={"file": ("data.xyz", content, "application/octet-stream")},
    )
    # Should fail with 400 (ValueError from loader) or 500
    assert r.status_code in (400, 500), f"Expected 400/500, got {r.status_code}: {r.text}"
    _print(f"Unsupported type response: {r.status_code} — {r.json()}")
    _print("✅ TEST 15 PASSED")


# ════════════════════════════════════════════════════════════════
#  PART C — File Uploads (local path)
# ════════════════════════════════════════════════════════════════


def test_16_upload_local_txt():
    """Test 16: POST /api/assets/uploads/file/local — upload TXT."""
    print("\n" + "=" * 70)
    print("  TEST 16: Upload local TXT file")
    print("=" * 70)

    coll = _coll("upload_txt")
    # Create collection first
    client.post("/api/assets/collections", json={"name": coll})

    r = client.post("/api/assets/uploads/file/local", json={
        "file_path": str(TXT_FILE),
        "collection_name": coll,
    })
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()
    assert data["success"] is True
    assert data["total_chunks"] >= 1
    assert len(data["document_ids"]) >= 1
    _print(f"Uploaded: chunks={data['total_chunks']}, ids={len(data['document_ids'])}")
    _print("✅ TEST 16 PASSED")


def test_17_upload_local_csv():
    """Test 17: POST /api/assets/uploads/file/local — upload CSV."""
    print("\n" + "=" * 70)
    print("  TEST 17: Upload local CSV file")
    print("=" * 70)

    coll = _coll("upload_csv")
    client.post("/api/assets/collections", json={"name": coll})

    r = client.post("/api/assets/uploads/file/local", json={
        "file_path": str(CSV_FILE),
        "collection_name": coll,
    })
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()
    assert data["success"] is True
    assert data["total_chunks"] >= 1
    _print(f"Uploaded: chunks={data['total_chunks']}, ids={len(data['document_ids'])}")
    _print("✅ TEST 17 PASSED")


def test_18_upload_local_json():
    """Test 18: POST /api/assets/uploads/file/local — upload JSON."""
    print("\n" + "=" * 70)
    print("  TEST 18: Upload local JSON file")
    print("=" * 70)

    coll = _coll("upload_json")
    client.post("/api/assets/collections", json={"name": coll})

    r = client.post("/api/assets/uploads/file/local", json={
        "file_path": str(JSON_FILE),
        "collection_name": coll,
    })
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()
    assert data["success"] is True
    assert data["total_chunks"] >= 1
    _print(f"Uploaded: chunks={data['total_chunks']}, ids={len(data['document_ids'])}")
    _print("✅ TEST 18 PASSED")


def test_19_upload_local_not_found():
    """Test 19: POST /api/assets/uploads/file/local — non-existent → 404."""
    print("\n" + "=" * 70)
    print("  TEST 19: Upload non-existent file (404)")
    print("=" * 70)

    r = client.post("/api/assets/uploads/file/local", json={
        "file_path": "/tmp/does_not_exist_xyz.txt",
        "collection_name": "irrelevant",
    })
    assert r.status_code == 404, f"Expected 404, got {r.status_code}: {r.text}"
    _print(f"404 response: {r.json()}")
    _print("✅ TEST 19 PASSED")


def test_20_verify_collection_stats_after_upload():
    """Test 20: GET collection stats — verify vector count after upload."""
    print("\n" + "=" * 70)
    print("  TEST 20: Verify collection stats after upload")
    print("=" * 70)

    coll = _coll("stats_check")
    client.post("/api/assets/collections", json={"name": coll})

    # Upload TXT
    r = client.post("/api/assets/uploads/file/local", json={
        "file_path": str(TXT_FILE),
        "collection_name": coll,
    })
    assert r.status_code == 200
    upload_data = r.json()
    expected_chunks = upload_data["total_chunks"]

    # Check stats
    r2 = client.get(f"/api/assets/collections/{coll}")
    assert r2.status_code == 200
    stats = r2.json()
    assert stats["vector_count"] == expected_chunks, \
        f"Expected {expected_chunks} vectors, got {stats['vector_count']}"
    assert stats["dimension"] == EMBEDDING_DIM
    _print(f"Collection '{coll}': {stats['vector_count']} vectors, dim={stats['dimension']}")
    _print("✅ TEST 20 PASSED")


# ════════════════════════════════════════════════════════════════
#  PART D — File Uploads (multipart)
# ════════════════════════════════════════════════════════════════


def test_21_upload_multipart_txt():
    """Test 21: POST /api/assets/uploads/file — multipart TXT upload."""
    print("\n" + "=" * 70)
    print("  TEST 21: Upload multipart TXT file")
    print("=" * 70)

    coll = _coll("multipart_txt")
    client.post("/api/assets/collections", json={"name": coll})

    with open(TXT_FILE, "rb") as f:
        r = client.post(
            "/api/assets/uploads/file",
            files={"file": (TXT_FILE.name, f, "text/plain")},
            data={"collection_name": coll},
        )
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()
    assert data["success"] is True
    assert data["total_chunks"] >= 1
    assert len(data["document_ids"]) >= 1
    # Source should show original filename, not a temp path
    assert data["source"] == TXT_FILE.name
    _print(f"Multipart upload: chunks={data['total_chunks']}, source={data['source']}")
    _print("✅ TEST 21 PASSED")


def test_22_upload_multipart_with_collection():
    """Test 22: POST /api/assets/uploads/file — multipart + explicit collection."""
    print("\n" + "=" * 70)
    print("  TEST 22: Upload multipart with collection_name")
    print("=" * 70)

    coll = _coll("multipart_coll")
    client.post("/api/assets/collections", json={"name": coll})

    with open(CSV_FILE, "rb") as f:
        r = client.post(
            "/api/assets/uploads/file",
            files={"file": (CSV_FILE.name, f, "text/csv")},
            data={"collection_name": coll},
        )
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()
    assert data["success"] is True
    assert data["metadata"].get("collection") == coll
    _print(f"Uploaded to collection '{coll}': chunks={data['total_chunks']}")

    # Verify via stats
    r2 = client.get(f"/api/assets/collections/{coll}")
    assert r2.status_code == 200
    _print(f"Stats: {r2.json()}")
    _print("✅ TEST 22 PASSED")


# ════════════════════════════════════════════════════════════════
#  PART E — Text Uploads
# ════════════════════════════════════════════════════════════════


def test_23_upload_text():
    """Test 23: POST /api/assets/uploads/text — single text."""
    print("\n" + "=" * 70)
    print("  TEST 23: Upload raw text")
    print("=" * 70)

    coll = _coll("text_upload")
    client.post("/api/assets/collections", json={"name": coll})

    r = client.post("/api/assets/uploads/text", json={
        "text": (
            "Garvit Batra is a software developer specializing in AI and machine learning. "
            "He has experience building RAG pipelines, LLM integrations, and full-stack apps."
        ),
        "collection_name": coll,
    })
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()
    assert data["success"] is True
    assert data["total_chunks"] >= 1
    _print(f"Text upload: chunks={data['total_chunks']}")
    _print("✅ TEST 23 PASSED")


def test_24_upload_text_with_metadata():
    """Test 24: POST /api/assets/uploads/text — with metadata + source."""
    print("\n" + "=" * 70)
    print("  TEST 24: Upload text with metadata")
    print("=" * 70)

    coll = _coll("text_meta")
    client.post("/api/assets/collections", json={"name": coll})

    r = client.post("/api/assets/uploads/text", json={
        "text": "Python is a versatile programming language used for web, AI, and automation.",
        "collection_name": coll,
        "metadata": {"category": "technology", "language": "en"},
        "source": "manual_input",
    })
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()
    assert data["success"] is True
    assert data["total_chunks"] >= 1
    _print(f"Text with metadata: chunks={data['total_chunks']}, meta={data['metadata']}")
    _print("✅ TEST 24 PASSED")


def test_25_upload_texts_batch():
    """Test 25: POST /api/assets/uploads/texts — batch texts."""
    print("\n" + "=" * 70)
    print("  TEST 25: Upload batch texts")
    print("=" * 70)

    coll = _coll("batch_texts")
    client.post("/api/assets/collections", json={"name": coll})

    texts = [
        "FastAPI is a modern web framework for building APIs with Python.",
        "React is a JavaScript library for building user interfaces.",
        "Qdrant is a vector similarity search engine for AI applications.",
    ]
    r = client.post("/api/assets/uploads/texts", json={
        "texts": texts,
        "collection_name": coll,
        "sources": ["doc_1", "doc_2", "doc_3"],
    })
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()
    assert data["success"] is True
    assert data["total_chunks"] >= 3  # at least 1 per text
    assert data["metadata"].get("document_count") == 3
    _print(f"Batch upload: chunks={data['total_chunks']}, docs={data['metadata'].get('document_count')}")

    # Verify count via stats
    r2 = client.get(f"/api/assets/collections/{coll}")
    assert r2.status_code == 200
    stats = r2.json()
    assert stats["vector_count"] == data["total_chunks"]
    _print(f"Verified: {stats['vector_count']} vectors in collection")
    _print("✅ TEST 25 PASSED")


# ════════════════════════════════════════════════════════════════
#  PART F — Directory Upload
# ════════════════════════════════════════════════════════════════


def test_26_upload_directory():
    """Test 26: POST /api/assets/uploads/directory/local — upload directory."""
    print("\n" + "=" * 70)
    print("  TEST 26: Upload local directory")
    print("=" * 70)

    coll = _coll("dir_upload")
    client.post("/api/assets/collections", json={"name": coll})

    r = client.post("/api/assets/uploads/directory/local", json={
        "directory_path": str(DATA_DIR),
        "collection_name": coll,
        "recursive": False,
        "extensions": ["txt", "csv", "json"],
    })
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()
    assert data["total_files"] >= 3  # txt, csv, json at minimum
    assert data["successful"] >= 3
    assert data["failed"] == 0
    _print(f"Directory upload: {data['total_files']} files, "
           f"{data['successful']} succeeded, {data['failed']} failed")
    for fr in data["results"]:
        _print(f"  {fr['source']}: chunks={fr['total_chunks']}, "
               f"success={fr['success']}")
    _print("✅ TEST 26 PASSED")


def test_27_upload_directory_filtered():
    """Test 27: POST /api/assets/uploads/directory/local — with extension filter."""
    print("\n" + "=" * 70)
    print("  TEST 27: Upload directory with extension filter")
    print("=" * 70)

    coll = _coll("dir_filtered")
    client.post("/api/assets/collections", json={"name": coll})

    r = client.post("/api/assets/uploads/directory/local", json={
        "directory_path": str(DATA_DIR),
        "collection_name": coll,
        "recursive": False,
        "extensions": ["txt"],  # only TXT files
    })
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()
    assert data["total_files"] >= 1
    # All files should be .txt
    for fr in data["results"]:
        if fr["source"]:
            assert fr["source"].endswith(".txt"), f"Non-txt file found: {fr['source']}"
    _print(f"Filtered upload: {data['total_files']} TXT files")
    _print("✅ TEST 27 PASSED")


def test_28_upload_directory_not_found():
    """Test 28: POST /api/assets/uploads/directory/local — non-existent → 404."""
    print("\n" + "=" * 70)
    print("  TEST 28: Upload non-existent directory (404)")
    print("=" * 70)

    r = client.post("/api/assets/uploads/directory/local", json={
        "directory_path": "/tmp/nonexistent_dir_xyz",
        "collection_name": "irrelevant",
    })
    assert r.status_code == 404, f"Expected 404, got {r.status_code}: {r.text}"
    _print(f"404 response: {r.json()}")
    _print("✅ TEST 28 PASSED")


# ════════════════════════════════════════════════════════════════
#  PART G — Full Dashboard Workflow
# ════════════════════════════════════════════════════════════════


def test_29_full_workflow_preview_then_upload():
    """
    Test 29: Full dashboard workflow.
    Create collection → preview file → upload file → verify stats.
    """
    print("\n" + "=" * 70)
    print("  TEST 29: Full workflow — preview → upload → verify")
    print("=" * 70)

    coll = _coll("workflow")

    # Step 1: Create collection
    r = client.post("/api/assets/collections", json={"name": coll})
    assert r.status_code == 201
    _print(f"Step 1: Created collection '{coll}'")

    # Step 2: Preview the file
    r = client.post("/api/assets/uploads/preview/local", json={
        "file_path": str(TXT_FILE),
    })
    assert r.status_code == 200
    preview = r.json()
    _print(f"Step 2: Preview — {preview['total_chunks']} chunks, "
           f"~{preview['estimated_tokens']} tokens")

    # Step 3: User sees preview and decides to upload
    r = client.post("/api/assets/uploads/file/local", json={
        "file_path": str(TXT_FILE),
        "collection_name": coll,
    })
    assert r.status_code == 200
    upload = r.json()
    assert upload["success"] is True
    # Chunk count from preview should match upload
    assert upload["total_chunks"] == preview["total_chunks"], \
        f"Preview chunks ({preview['total_chunks']}) != upload chunks ({upload['total_chunks']})"
    _print(f"Step 3: Uploaded — {upload['total_chunks']} chunks, "
           f"{len(upload['document_ids'])} IDs")

    # Step 4: Verify stats
    r = client.get(f"/api/assets/collections/{coll}")
    assert r.status_code == 200
    stats = r.json()
    assert stats["vector_count"] == upload["total_chunks"]
    assert stats["dimension"] == EMBEDDING_DIM
    _print(f"Step 4: Verified — {stats['vector_count']} vectors, "
           f"dim={stats['dimension']}")

    _print("✅ TEST 29 PASSED")


def test_30_full_workflow_text_upload_then_delete():
    """
    Test 30: Full workflow.
    Upload text → verify stats → delete collection → verify deleted.
    """
    print("\n" + "=" * 70)
    print("  TEST 30: Full workflow — text upload → stats → delete")
    print("=" * 70)

    coll = _coll("text_workflow")

    # Step 1: Upload text (auto-creates collection)
    r = client.post("/api/assets/uploads/text", json={
        "text": (
            "Machine learning is a subset of artificial intelligence that "
            "provides systems the ability to learn and improve from experience. "
            "Deep learning is a sub-field of machine learning that uses neural networks."
        ),
        "collection_name": coll,
        "source": "knowledge_base",
    })
    assert r.status_code == 200
    upload = r.json()
    assert upload["success"] is True
    _print(f"Step 1: Uploaded text — {upload['total_chunks']} chunks")

    # Step 2: Verify stats
    r = client.get(f"/api/assets/collections/{coll}")
    assert r.status_code == 200
    stats = r.json()
    assert stats["vector_count"] == upload["total_chunks"]
    _print(f"Step 2: Stats — {stats['vector_count']} vectors")

    # Step 3: Verify it appears in collection list
    r = client.get("/api/assets/collections")
    assert r.status_code == 200
    listing = r.json()
    coll_names = [c["collection"] for c in listing["collections"]]
    assert coll in coll_names, f"'{coll}' not in collection list: {coll_names}"
    _print(f"Step 3: Collection '{coll}' visible in list")

    # Step 4: Delete collection
    r = client.delete(f"/api/assets/collections/{coll}")
    assert r.status_code == 200
    assert r.json()["deleted"] is True
    _print(f"Step 4: Deleted collection '{coll}'")

    # Step 5: Verify it's gone
    r = client.get(f"/api/assets/collections/{coll}")
    assert r.status_code == 404
    _print("Step 5: Confirmed deleted (404)")

    _print("✅ TEST 30 PASSED")


# ── Main Runner ──────────────────────────────────────────────────

def main():
    """Run all tests in order."""
    print("\n" + "=" * 70)
    print("  Asset Upload Dashboard — E2E API Tests")
    print(f"  Run ID: {RUN_ID}")
    print(f"  Data dir: {DATA_DIR}")
    print("=" * 70)

    # Check for Cohere API key
    if not os.environ.get("COHERE_API_KEY") and not os.environ.get("COHERE_API_KEYS"):
        print("\n  ❌ COHERE_API_KEY not set — these tests require real embeddings.")
        print("     Set COHERE_API_KEY in .env and retry.\n")
        sys.exit(1)

    tests = [
        # Part A — Collection Management
        test_01_list_collections_empty,
        test_02_create_collection,
        test_03_create_collection_duplicate,
        test_04_get_collection_stats,
        test_05_get_collection_not_found,
        test_06_delete_collection,
        test_07_delete_collection_not_found,
        # Part B — Upload Previews
        test_08_supported_types,
        test_09_preview_local_txt,
        test_10_preview_local_csv,
        test_11_preview_local_json,
        test_12_preview_local_not_found,
        test_13_preview_multipart_txt,
        test_14_preview_multipart_csv,
        test_15_preview_multipart_unsupported,
        # Part C — File Uploads (local path)
        test_16_upload_local_txt,
        test_17_upload_local_csv,
        test_18_upload_local_json,
        test_19_upload_local_not_found,
        test_20_verify_collection_stats_after_upload,
        # Part D — File Uploads (multipart)
        test_21_upload_multipart_txt,
        test_22_upload_multipart_with_collection,
        # Part E — Text Uploads
        test_23_upload_text,
        test_24_upload_text_with_metadata,
        test_25_upload_texts_batch,
        # Part F — Directory Upload
        test_26_upload_directory,
        test_27_upload_directory_filtered,
        test_28_upload_directory_not_found,
        # Part G — Full Dashboard Workflow
        test_29_full_workflow_preview_then_upload,
        test_30_full_workflow_text_upload_then_delete,
    ]

    passed = 0
    failed = 0
    errors = []

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((test_fn.__name__, str(e)))
            print(f"  ❌ {test_fn.__name__} FAILED: {e}")

    print("\n" + "=" * 70)
    print(f"  Results: {passed} passed, {failed} failed out of {len(tests)}")
    if errors:
        print("\n  Failures:")
        for name, err in errors:
            print(f"    ❌ {name}: {err}")
    else:
        print("  ✅ All tests passed!")
    print("=" * 70 + "\n")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
