"""
End-to-end tests for the file-level admin endpoints.

Exercises the three new endpoints that let the admin panel manage
individual files inside a collection:

  1. GET    /api/assets/collections/{name}/files          (list files)
  2. GET    /api/assets/collections/{name}/files/chunks   (get a file's chunks)
  3. DELETE /api/assets/collections/{name}/files          (delete one file)

Also verifies that the canonical payload fields (source, file_type,
uploaded_at, file_size_bytes) are stamped at upload time.

Requires (.env):
    COHERE_API_KEY  — Cohere embed-english-v3.0

Run:
    PYTHONPATH=. python tests/test_file_management_e2e.py
    PYTHONPATH=. python -m pytest tests/test_file_management_e2e.py -v --tb=short
"""

from __future__ import annotations

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
from AssetUploadService.services.dashboard_service import DashboardService
from AssetUploadService.Server.routes.collections import router as collections_router
from AssetUploadService.Server.routes.uploads import router as uploads_router

EMBEDDING_DIM = 1024
RUN_ID = uuid.uuid4().hex[:8]

DATA_DIR = Path(__file__).parent / "document_loaders_tests"
TXT_FILE = DATA_DIR / "my all details.txt"
CSV_FILE = DATA_DIR / "Personal_TrainingData.csv"

_test_service: DashboardService | None = None


def _build_test_service() -> DashboardService:
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
    service = _build_test_service()
    import AssetUploadService.Server.routes.collections as coll_mod
    import AssetUploadService.Server.routes.uploads as upl_mod
    coll_mod._get_dashboard_service = lambda: service
    upl_mod._get_dashboard_service = lambda: service

    app = FastAPI()
    app.include_router(collections_router, prefix="/api")
    app.include_router(uploads_router, prefix="/api")
    return app


_app = _build_test_app()
client = TestClient(_app)

COLLECTION = f"files_test_{RUN_ID}"


def _print(msg: str):
    print(f"  {msg}")


def test_01_setup_upload_two_files():
    """Upload two distinct files into one collection."""
    print("\n" + "=" * 70)
    print("  TEST 1: Upload two files into a collection")
    print("=" * 70)

    # Upload TXT
    r = client.post(
        "/api/assets/uploads/file/local",
        json={"file_path": str(TXT_FILE), "collection_name": COLLECTION},
    )
    assert r.status_code == 200, f"{r.status_code}: {r.text}"
    assert r.json()["success"] is True
    _print(f"Uploaded TXT: {r.json()['total_chunks']} chunks")

    # Upload CSV
    r = client.post(
        "/api/assets/uploads/file/local",
        json={"file_path": str(CSV_FILE), "collection_name": COLLECTION},
    )
    assert r.status_code == 200, f"{r.status_code}: {r.text}"
    assert r.json()["success"] is True
    _print(f"Uploaded CSV: {r.json()['total_chunks']} chunks")
    _print("✅ TEST 1 PASSED")


def test_02_list_files():
    """List files — should return both files with metadata."""
    print("\n" + "=" * 70)
    print("  TEST 2: List files in collection")
    print("=" * 70)

    r = client.get(f"/api/assets/collections/{COLLECTION}/files")
    assert r.status_code == 200, f"{r.status_code}: {r.text}"
    data = r.json()

    assert data["total"] == 2, f"Expected 2 files, got {data['total']}"
    sources = {f["source"] for f in data["files"]}
    assert str(TXT_FILE) in sources
    assert str(CSV_FILE) in sources

    for f in data["files"]:
        _print(
            f"source={Path(f['source']).name} chunks={f['chunk_count']} "
            f"type={f['file_type']} uploaded_at={f['uploaded_at']} "
            f"size={f['file_size_bytes']}"
        )
        # Metadata gaps that were requested:
        assert f["chunk_count"] > 0
        assert f["file_type"] is not None, "file_type must be stamped"
        assert f["uploaded_at"] is not None, "uploaded_at must be stamped"
        assert f["file_size_bytes"] is not None, "file_size_bytes must be stamped"

    _print("✅ TEST 2 PASSED")


def test_03_list_files_404_for_missing_collection():
    """Listing files for a missing collection returns 404."""
    print("\n" + "=" * 70)
    print("  TEST 3: List files — missing collection → 404")
    print("=" * 70)

    r = client.get(f"/api/assets/collections/does_not_exist_{RUN_ID}/files")
    assert r.status_code == 404, f"{r.status_code}: {r.text}"
    _print("✅ TEST 3 PASSED")


def test_04_get_file_chunks():
    """Get the chunks of one file, ordered by chunk_index."""
    print("\n" + "=" * 70)
    print("  TEST 4: Get chunks of a single file")
    print("=" * 70)

    r = client.get(
        f"/api/assets/collections/{COLLECTION}/files/chunks",
        params={"source": str(CSV_FILE)},
    )
    assert r.status_code == 200, f"{r.status_code}: {r.text}"
    data = r.json()

    assert data["source"] == str(CSV_FILE)
    assert data["total"] > 0
    assert len(data["chunks"]) == data["total"]

    # Ordered by chunk_index
    indices = [c["chunk_index"] for c in data["chunks"]]
    assert indices == sorted(indices), "Chunks must be ordered by chunk_index"

    first = data["chunks"][0]
    assert "id" in first and "content" in first and "metadata" in first
    _print(f"Got {data['total']} chunks; first content: {first['content'][:60]!r}")
    _print("✅ TEST 4 PASSED")


def test_05_get_file_chunks_404_for_missing_source():
    """Requesting chunks for an unknown source returns 404."""
    print("\n" + "=" * 70)
    print("  TEST 5: Get chunks — unknown source → 404")
    print("=" * 70)

    r = client.get(
        f"/api/assets/collections/{COLLECTION}/files/chunks",
        params={"source": "no_such_file.pdf"},
    )
    assert r.status_code == 404, f"{r.status_code}: {r.text}"
    _print("✅ TEST 5 PASSED")


def test_06_delete_single_file():
    """Delete one file; the other remains intact."""
    print("\n" + "=" * 70)
    print("  TEST 6: Delete a single file")
    print("=" * 70)

    # Count chunks of the TXT file first
    r = client.get(
        f"/api/assets/collections/{COLLECTION}/files/chunks",
        params={"source": str(TXT_FILE)},
    )
    assert r.status_code == 200
    txt_chunks = r.json()["total"]

    # Delete the TXT file
    r = client.delete(
        f"/api/assets/collections/{COLLECTION}/files",
        params={"source": str(TXT_FILE)},
    )
    assert r.status_code == 200, f"{r.status_code}: {r.text}"
    data = r.json()
    assert data["success"] is True
    assert data["deleted_chunks"] == txt_chunks, (
        f"Expected {txt_chunks} deleted, got {data['deleted_chunks']}"
    )
    _print(f"Deleted {data['deleted_chunks']} chunks of {Path(str(TXT_FILE)).name}")

    # Verify only the CSV file remains
    r = client.get(f"/api/assets/collections/{COLLECTION}/files")
    assert r.status_code == 200
    data = r.json()
    assert data["total"] == 1, f"Expected 1 file left, got {data['total']}"
    assert data["files"][0]["source"] == str(CSV_FILE)
    _print("✅ TEST 6 PASSED — other file intact")


def test_07_delete_missing_file_is_idempotent():
    """Deleting an already-deleted file returns success with 0 chunks."""
    print("\n" + "=" * 70)
    print("  TEST 7: Delete already-deleted file → idempotent")
    print("=" * 70)

    r = client.delete(
        f"/api/assets/collections/{COLLECTION}/files",
        params={"source": str(TXT_FILE)},
    )
    assert r.status_code == 200, f"{r.status_code}: {r.text}"
    data = r.json()
    assert data["success"] is True
    assert data["deleted_chunks"] == 0
    _print("✅ TEST 7 PASSED")


def test_08_text_upload_has_canonical_fields():
    """Raw text upload should also be listable with canonical fields."""
    print("\n" + "=" * 70)
    print("  TEST 8: Raw text upload is listed with metadata")
    print("=" * 70)

    r = client.post(
        "/api/assets/uploads/text",
        json={
            "text": "A standalone note for the admin panel.",
            "collection_name": COLLECTION,
            "source": "note_001",
        },
    )
    assert r.status_code == 200, f"{r.status_code}: {r.text}"

    r = client.get(f"/api/assets/collections/{COLLECTION}/files")
    assert r.status_code == 200
    files = {f["source"]: f for f in r.json()["files"]}
    assert "note_001" in files, "Text source should appear in file list"
    note = files["note_001"]
    assert note["file_type"] == "text"
    assert note["uploaded_at"] is not None
    _print(f"note_001: type={note['file_type']} uploaded_at={note['uploaded_at']}")
    _print("✅ TEST 8 PASSED")


def test_09_file_size_bytes_populated():
    """Every listed file reports a non-null, positive file_size_bytes."""
    print("\n" + "=" * 70)
    print("  TEST 9: file_size_bytes is populated for all files")
    print("=" * 70)

    r = client.get(f"/api/assets/collections/{COLLECTION}/files")
    assert r.status_code == 200, f"{r.status_code}: {r.text}"
    files = r.json()["files"]
    assert files, "expected at least one file"
    for f in files:
        assert f["file_size_bytes"] is not None, (
            f"file_size_bytes is null for {f['source']}"
        )
        assert f["file_size_bytes"] > 0, (
            f"file_size_bytes should be > 0 for {f['source']}"
        )
        _print(f"{Path(f['source']).name}: {f['file_size_bytes']} bytes")
    _print("✅ TEST 9 PASSED")


def test_10_multipart_upload_with_tags_and_metadata():
    """Multipart /file accepts tags + metadata; tags surface in the file list."""
    print("\n" + "=" * 70)
    print("  TEST 10: Multipart upload with tags + metadata")
    print("=" * 70)

    tag_collection = f"tags_test_{RUN_ID}"
    with open(TXT_FILE, "rb") as fh:
        r = client.post(
            "/api/assets/uploads/file",
            files={"file": ("my all details.txt", fh, "text/plain")},
            data={
                "collection_name": tag_collection,
                "tags": '["projects", "resume"]',
                "metadata": '{"category": "personal"}',
            },
        )
    assert r.status_code == 200, f"{r.status_code}: {r.text}"
    assert r.json()["success"] is True
    _print(f"Uploaded with tags; {r.json()['total_chunks']} chunks")

    # Tags surface in the file list
    r = client.get(f"/api/assets/collections/{tag_collection}/files")
    assert r.status_code == 200, f"{r.status_code}: {r.text}"
    files = r.json()["files"]
    assert len(files) == 1, f"expected 1 file, got {len(files)}"
    f = files[0]
    assert f["tags"] == ["projects", "resume"], f"unexpected tags: {f['tags']}"
    assert f["file_size_bytes"] and f["file_size_bytes"] > 0
    assert f["source"] == "my all details.txt", (
        f"source should be original filename, got {f['source']}"
    )
    _print(f"File list tags={f['tags']} size={f['file_size_bytes']}")

    # Custom metadata persists on the chunks
    r = client.get(
        f"/api/assets/collections/{tag_collection}/files/chunks",
        params={"source": "my all details.txt"},
    )
    assert r.status_code == 200, f"{r.status_code}: {r.text}"
    first_meta = r.json()["chunks"][0]["metadata"]
    assert first_meta.get("category") == "personal", (
        f"custom metadata not persisted: {first_meta}"
    )
    assert first_meta.get("tags") == ["projects", "resume"]
    _print("✅ TEST 10 PASSED — tags + metadata persisted")


def test_11_comma_separated_tags():
    """Tags can also be supplied as a comma-separated string."""
    print("\n" + "=" * 70)
    print("  TEST 11: Comma-separated tags")
    print("=" * 70)

    coll = f"csv_tags_{RUN_ID}"
    with open(CSV_FILE, "rb") as fh:
        r = client.post(
            "/api/assets/uploads/file",
            files={"file": ("Personal_TrainingData.csv", fh, "text/csv")},
            data={"collection_name": coll, "tags": "training, data, csv"},
        )
    assert r.status_code == 200, f"{r.status_code}: {r.text}"

    r = client.get(f"/api/assets/collections/{coll}/files")
    assert r.status_code == 200
    f = r.json()["files"][0]
    assert f["tags"] == ["training", "data", "csv"], f"got {f['tags']}"
    _print(f"✅ TEST 11 PASSED — comma tags parsed: {f['tags']}")


def main():
    tests = [
        test_01_setup_upload_two_files,
        test_02_list_files,
        test_03_list_files_404_for_missing_collection,
        test_04_get_file_chunks,
        test_05_get_file_chunks_404_for_missing_source,
        test_06_delete_single_file,
        test_07_delete_missing_file_is_idempotent,
        test_08_text_upload_has_canonical_fields,
        test_09_file_size_bytes_populated,
        test_10_multipart_upload_with_tags_and_metadata,
        test_11_comma_separated_tags,
    ]
    passed = 0
    for t in tests:
        t()
        passed += 1
    print("\n" + "=" * 70)
    print(f"  ✅ ALL {passed}/{len(tests)} FILE-MANAGEMENT TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
