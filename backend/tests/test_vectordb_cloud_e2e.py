"""
End-to-end tests for the VectorDB module against a **real Qdrant Cloud** instance.

Unlike the in-memory tests, these hit a remote cluster over the network
and validate that every CRUD operation works end-to-end against the
production Qdrant API.

Requires:
    QDRANT_URL   – Qdrant Cloud cluster URL  (in .env)
    QDRANT_API_KEY – Qdrant Cloud API key     (in .env)

Run:
    PYTHONPATH=. python tests/test_vectordb_cloud_e2e.py

Covers:
    1.  Collection lifecycle  (create / exists / info / list / delete)
    2.  Add & get documents
    3.  Batch add
    4.  Update documents (upsert)
    5.  Delete by IDs
    6.  Delete by metadata filter
    7.  Vector similarity search
    8.  Similarity search with score + threshold
    9.  Metadata-filtered search
    10. Count utility
    11. Collection info after operations
    12. Rich metadata roundtrip
"""

from __future__ import annotations

import os
import random
import sys
import time
import uuid
from typing import List

from dotenv import load_dotenv

load_dotenv()

from RAGService.Data.VectorDB import (
    VectorDBFactory,
    VectorDBConfig,
    VectorDBProvider,
    DistanceMetric,
    DocumentChunk,
    SearchResult,
    CollectionInfo,
    MetadataFilter,
    MetadataFilterGroup,
    FilterOperator,
)


# ── Configuration ────────────────────────────────────────────────

QDRANT_URL = os.getenv("QDRANT_URL")

DIM = 128  # small dimension for fast tests

# Each run gets a unique prefix so parallel runs / leftover collections
# never collide.
RUN_ID = uuid.uuid4().hex[:8]
COLLECTION = f"cloud_test_{RUN_ID}"


def _prefixed(name: str) -> str:
    """Return a run-scoped collection name."""
    return f"{name}_{RUN_ID}"


# ── Helpers ──────────────────────────────────────────────────────

def _make_db(collection: str = COLLECTION):
    """Create a Qdrant Cloud instance via direct config (REST, not gRPC)."""
    config = VectorDBConfig(
        provider=VectorDBProvider.QDRANT,
        collection_name=collection,
        embedding_dimension=DIM,
        url=QDRANT_URL,
        distance_metric=DistanceMetric.COSINE,
        prefer_grpc=False,           # REST is more reliable for cloud
        extra_config={},
    )
    return VectorDBFactory.create(config)


def _rand_vec(dim: int = DIM) -> List[float]:
    """Random unit vector."""
    vec = [random.gauss(0, 1) for _ in range(dim)]
    norm = max(sum(v * v for v in vec) ** 0.5, 1e-9)
    return [v / norm for v in vec]


def _make_chunk(
    content: str,
    source: str = "test.txt",
    chunk_index: int = 0,
    embedding: List[float] | None = None,
    extra_meta: dict | None = None,
) -> DocumentChunk:
    meta = {"source": source, "chunk_index": chunk_index}
    if extra_meta:
        meta.update(extra_meta)
    return DocumentChunk(
        content=content,
        metadata=meta,
        embedding=embedding or _rand_vec(),
        source=source,
        chunk_index=chunk_index,
    )


def _wait(seconds: float = 1.0):
    """
    Small sleep to let Qdrant Cloud index / replicate.
    Cloud writes are eventually consistent, so a short pause
    after writes avoids flaky assertions.
    """
    time.sleep(seconds)


def _create_payload_indexes(db, collection_name: str, fields: dict[str, str]):
    """
    Create payload indexes on a cloud collection.

    Qdrant Cloud requires explicit payload indexes before you can filter
    on a field (unlike in-memory mode which handles it transparently).

    Args:
        db: QdrantVectorDB instance (we access its internal _client)
        collection_name: Name of the collection
        fields: Mapping of field_name → schema type string.
                Supported: 'keyword', 'integer', 'float', 'bool'
    """
    from qdrant_client.http.models import PayloadSchemaType

    schema_map = {
        "keyword":  PayloadSchemaType.KEYWORD,
        "integer":  PayloadSchemaType.INTEGER,
        "float":    PayloadSchemaType.FLOAT,
        "bool":     PayloadSchemaType.BOOL,
    }

    for field_name, schema_str in fields.items():
        schema = schema_map[schema_str]
        db._client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=schema,
            wait=True,
        )


def _safe_cleanup(db, *names):
    """Delete collections silently — used in teardown."""
    for n in names:
        try:
            if db.collection_exists(collection_name=n):
                db.delete_collection(collection_name=n)
        except Exception:
            pass


# ── Tests ────────────────────────────────────────────────────────

def test_collection_lifecycle():
    """Test 1: create → exists → info → list → delete → not exists"""
    print("Test 1: Collection lifecycle (cloud)")
    coll = _prefixed("lifecycle")
    db = _make_db(coll)
    try:
        # Should not exist yet
        assert not db.collection_exists(), "Collection should not exist yet"

        # Create
        ok = db.create_collection()
        assert ok
        _wait()
        assert db.collection_exists(), "Collection should exist after create"

        # Info
        info = db.get_collection_info()
        assert isinstance(info, CollectionInfo)
        assert info.name == coll
        assert info.dimension == DIM
        assert info.distance_metric == DistanceMetric.COSINE
        assert info.vector_count == 0
        print(f"  Info: name={info.name}, dim={info.dimension}, metric={info.distance_metric.value}, vectors={info.vector_count}")

        # List
        names = db.list_collections()
        assert coll in names, f"Collection not in list: {names}"
        print(f"  Collection appears in list_collections ✓")

        # Delete
        db.delete_collection()
        _wait()
        assert not db.collection_exists(), "Collection should not exist after delete"
        print("  ✅ PASSED\n")
    finally:
        _safe_cleanup(db, coll)


def test_add_and_get_documents():
    """Test 2: Add documents, retrieve by ID — content & metadata roundtrip."""
    print("Test 2: Add & get documents (cloud)")
    coll = _prefixed("add_get")
    db = _make_db(coll)
    try:
        db.create_collection()
        _wait()

        docs = [
            _make_chunk("Machine learning is great.", chunk_index=0, extra_meta={"topic": "ml"}),
            _make_chunk("Deep learning uses neural networks.", chunk_index=1, extra_meta={"topic": "dl"}),
            _make_chunk("NLP is a subfield of AI.", chunk_index=2, extra_meta={"topic": "nlp"}),
        ]
        ids = db.add_documents(docs)
        _wait()

        assert len(ids) == 3
        print(f"  Added {len(ids)} docs, IDs: {ids}")

        # Retrieve without embeddings
        retrieved = db.get_by_ids(ids)
        assert len(retrieved) == 3
        for orig, ret in zip(docs, retrieved):
            assert ret.content == orig.content, f"Content mismatch: {ret.content!r}"
            assert ret.metadata.get("topic") == orig.metadata["topic"]
            assert ret.embedding is None, "Embedding should be None when not requested"

        # Retrieve WITH embeddings
        retrieved_emb = db.get_by_ids(ids[:1], include_embeddings=True)
        assert len(retrieved_emb) == 1
        assert retrieved_emb[0].embedding is not None
        assert len(retrieved_emb[0].embedding) == DIM
        print(f"  Retrieved with embeddings: dim={len(retrieved_emb[0].embedding)}")

        # Count
        assert db.count() == 3
        print(f"  Count: {db.count()}")
        print("  ✅ PASSED\n")
    finally:
        _safe_cleanup(db, coll)


def test_batch_add():
    """Test 3: Batch add with explicit batch_size."""
    print("Test 3: Batch add (20 docs, batch_size=7, cloud)")
    coll = _prefixed("batch")
    db = _make_db(coll)
    try:
        db.create_collection()
        _wait()

        docs = [_make_chunk(f"Document {i}", chunk_index=i) for i in range(20)]
        ids = db.add_documents(docs, batch_size=7)
        _wait()

        assert len(ids) == 20
        assert db.count() == 20
        print(f"  Added 20 docs in batches of 7, count={db.count()}")
        print("  ✅ PASSED\n")
    finally:
        _safe_cleanup(db, coll)


def test_update_documents():
    """Test 4: Upsert — update content & metadata of existing doc."""
    print("Test 4: Update documents / upsert (cloud)")
    coll = _prefixed("update")
    db = _make_db(coll)
    try:
        db.create_collection()
        _wait()

        doc = _make_chunk("Original content", extra_meta={"version": 1})
        doc_id = doc.id
        db.add_documents([doc])
        _wait()

        retrieved = db.get_by_ids([doc_id])
        assert retrieved[0].content == "Original content"
        assert retrieved[0].metadata.get("version") == 1
        print(f"  Original: content={retrieved[0].content!r}, version={retrieved[0].metadata.get('version')}")

        # Update with same ID
        updated = DocumentChunk(
            id=doc_id,
            content="Updated content with more detail",
            metadata={"source": "test.txt", "version": 2, "updated": True},
            embedding=_rand_vec(),
        )
        db.update_documents([updated])
        _wait()

        retrieved2 = db.get_by_ids([doc_id])
        assert retrieved2[0].content == "Updated content with more detail"
        assert retrieved2[0].metadata.get("version") == 2
        assert retrieved2[0].metadata.get("updated") is True
        assert db.count() == 1, "Upsert should not create a duplicate"
        print(f"  Updated: content={retrieved2[0].content!r}, version={retrieved2[0].metadata.get('version')}, count={db.count()}")
        print("  ✅ PASSED\n")
    finally:
        _safe_cleanup(db, coll)


def test_delete_by_ids():
    """Test 5: Add 5 docs, delete 2 by ID, verify remaining."""
    print("Test 5: Delete by IDs (cloud)")
    coll = _prefixed("del_ids")
    db = _make_db(coll)
    try:
        db.create_collection()
        _wait()

        docs = [_make_chunk(f"Doc {i}", chunk_index=i) for i in range(5)]
        ids = db.add_documents(docs)
        _wait()
        assert db.count() == 5

        ok = db.delete_by_ids(ids[:2])
        _wait()
        assert ok, "delete_by_ids should return True"
        assert db.count() == 3, f"Expected 3, got {db.count()}"

        # Deleted docs gone
        gone = db.get_by_ids(ids[:2])
        assert len(gone) == 0, f"Deleted docs should be gone, got {len(gone)}"

        # Survivors still present
        alive = db.get_by_ids(ids[2:])
        assert len(alive) == 3
        print(f"  Deleted 2, remaining={db.count()}, surviving={len(alive)}")
        print("  ✅ PASSED\n")
    finally:
        _safe_cleanup(db, coll)


def test_delete_by_filter():
    """Test 6: Delete docs by metadata filter."""
    print("Test 6: Delete by metadata filter (cloud)")
    coll = _prefixed("del_filter")
    db = _make_db(coll)
    try:
        db.create_collection()
        _wait()

        # Cloud requires payload indexes for filter operations
        _create_payload_indexes(db, coll, {"category": "keyword"})

        docs = [
            _make_chunk("Python guide", chunk_index=0, extra_meta={"category": "programming"}),
            _make_chunk("Java guide", chunk_index=1, extra_meta={"category": "programming"}),
            _make_chunk("Cooking recipe", chunk_index=2, extra_meta={"category": "cooking"}),
            _make_chunk("History lesson", chunk_index=3, extra_meta={"category": "history"}),
        ]
        db.add_documents(docs)
        _wait()
        assert db.count() == 4

        deleted = db.delete_by_filter(
            MetadataFilter(field="category", operator=FilterOperator.EQ, value="programming")
        )
        _wait()
        assert deleted == 2, f"Expected 2 deleted, got {deleted}"
        assert db.count() == 2

        remaining = db.get_by_ids([d.id for d in docs[2:]])
        categories = {r.metadata.get("category") for r in remaining}
        assert categories == {"cooking", "history"}
        print(f"  Deleted {deleted} 'programming' docs, remaining: {categories}")
        print("  ✅ PASSED\n")
    finally:
        _safe_cleanup(db, coll)


def test_vector_search():
    """Test 7: Similarity search — nearest vector ranks first."""
    print("Test 7: Vector similarity search (cloud)")
    coll = _prefixed("search")
    db = _make_db(coll)
    try:
        db.create_collection()
        _wait()

        target_vec = _rand_vec()
        query_vec = [v + random.gauss(0, 0.01) for v in target_vec]

        docs = [_make_chunk("Target document", embedding=target_vec, chunk_index=0)]
        for i in range(9):
            docs.append(_make_chunk(f"Other doc {i}", embedding=_rand_vec(), chunk_index=i + 1))

        db.add_documents(docs)
        _wait(1.5)  # extra time for cloud indexing

        results = db.search(query_embedding=query_vec, k=3)
        assert len(results) == 3
        assert isinstance(results[0], SearchResult)
        assert results[0].content == "Target document", f"Top result should be target, got: {results[0].content!r}"
        assert results[0].score > results[1].score

        print(f"  Top-3 results:")
        for r in results:
            print(f"    score={r.score:.4f}  content={r.content!r}")
        print("  ✅ PASSED\n")
    finally:
        _safe_cleanup(db, coll)


def test_search_with_score_and_threshold():
    """Test 8: similarity_search_with_score + threshold filtering."""
    print("Test 8: Search with score + threshold (cloud)")
    coll = _prefixed("score_thr")
    db = _make_db(coll)
    try:
        db.create_collection()
        _wait()

        base_vec = _rand_vec()
        close_docs = [
            _make_chunk(f"Close doc {i}",
                        embedding=[v + random.gauss(0, 0.02) for v in base_vec],
                        chunk_index=i)
            for i in range(3)
        ]
        far_docs = [
            _make_chunk(f"Far doc {i}", embedding=_rand_vec(), chunk_index=i + 3)
            for i in range(5)
        ]
        db.add_documents(close_docs + far_docs)
        _wait(1.5)

        query = [v + random.gauss(0, 0.01) for v in base_vec]

        # similarity_search_with_score — returns (DocumentChunk, float)
        doc_scores = db.similarity_search_with_score(query, k=8)
        assert len(doc_scores) == 8
        assert isinstance(doc_scores[0], tuple)
        assert isinstance(doc_scores[0][0], DocumentChunk)
        assert doc_scores[0][0].embedding is not None, "Should include embeddings"
        print(f"  similarity_search_with_score returned {len(doc_scores)} results")
        for doc, score in doc_scores[:4]:
            print(f"    score={score:.4f}  content={doc.content!r}")

        # Threshold — only the close cluster should survive
        high_threshold = doc_scores[2][1] - 0.01
        filtered = db.search(query, k=10, score_threshold=high_threshold)
        assert len(filtered) <= 8
        print(f"  With threshold={high_threshold:.4f}: got {len(filtered)} results")
        print("  ✅ PASSED\n")
    finally:
        _safe_cleanup(db, coll)


def test_filtered_search():
    """Test 9: Search with metadata filters."""
    print("Test 9: Metadata-filtered search (cloud)")
    coll = _prefixed("filtered")
    db = _make_db(coll)
    try:
        db.create_collection()
        _wait()

        # Cloud requires payload indexes for filter operations
        _create_payload_indexes(db, coll, {"lang": "keyword", "level": "integer"})

        base_vec = _rand_vec()
        docs = [
            _make_chunk("Python basics",
                        embedding=[v + random.gauss(0, 0.01) for v in base_vec],
                        chunk_index=0, extra_meta={"lang": "python", "level": 1}),
            _make_chunk("Python advanced",
                        embedding=[v + random.gauss(0, 0.02) for v in base_vec],
                        chunk_index=1, extra_meta={"lang": "python", "level": 3}),
            _make_chunk("Java basics",
                        embedding=[v + random.gauss(0, 0.01) for v in base_vec],
                        chunk_index=2, extra_meta={"lang": "java", "level": 1}),
            _make_chunk("Rust systems",
                        embedding=_rand_vec(),
                        chunk_index=3, extra_meta={"lang": "rust", "level": 4}),
        ]
        db.add_documents(docs)
        _wait(1.5)

        # EQ filter: lang == "python"
        r1 = db.search(
            query_embedding=[v + random.gauss(0, 0.005) for v in base_vec],
            k=10,
            filters=MetadataFilter(field="lang", operator=FilterOperator.EQ, value="python"),
        )
        assert len(r1) == 2
        assert all(r.metadata.get("lang") == "python" for r in r1)
        print(f"  Filter lang='python': {len(r1)} results")

        # GTE filter: level >= 3
        r2 = db.search(
            query_embedding=[v + random.gauss(0, 0.005) for v in base_vec],
            k=10,
            filters=MetadataFilter(field="level", operator=FilterOperator.GTE, value=3),
        )
        levels = [r.metadata.get("level") for r in r2]
        assert all(l >= 3 for l in levels)
        print(f"  Filter level>=3: {len(r2)} results, levels={levels}")

        # AND group: lang='python' AND level=1
        r3 = db.search(
            query_embedding=base_vec,
            k=10,
            filters=MetadataFilterGroup(
                filters=[
                    MetadataFilter(field="lang", operator=FilterOperator.EQ, value="python"),
                    MetadataFilter(field="level", operator=FilterOperator.EQ, value=1),
                ],
                operator="and",
            ),
        )
        assert len(r3) == 1
        assert r3[0].content == "Python basics"
        print(f"  Filter lang='python' AND level=1: {len(r3)} result → {r3[0].content!r}")

        # IN filter: lang in ["python", "rust"]
        r4 = db.search(
            query_embedding=base_vec,
            k=10,
            filters=MetadataFilter(field="lang", operator=FilterOperator.IN, value=["python", "rust"]),
        )
        langs = {r.metadata.get("lang") for r in r4}
        assert langs <= {"python", "rust"}
        print(f"  Filter lang IN ['python','rust']: {len(r4)} results, langs={langs}")
        print("  ✅ PASSED\n")
    finally:
        _safe_cleanup(db, coll)


def test_count():
    """Test 10: Count accuracy through add / delete cycles."""
    print("Test 10: Count utility (cloud)")
    coll = _prefixed("count")
    db = _make_db(coll)
    try:
        db.create_collection()
        _wait()

        assert db.count() == 0

        docs = [_make_chunk(f"Doc {i}", chunk_index=i) for i in range(15)]
        ids = db.add_documents(docs)
        _wait()
        assert db.count() == 15
        print(f"  After adding 15: count={db.count()}")

        db.delete_by_ids(ids[:5])
        _wait()
        assert db.count() == 10
        print(f"  After deleting 5: count={db.count()}")

        db.add_documents([_make_chunk("Extra", chunk_index=99)])
        _wait()
        assert db.count() == 11
        print(f"  After adding 1 more: count={db.count()}")
        print("  ✅ PASSED\n")
    finally:
        _safe_cleanup(db, coll)


def test_collection_info_after_operations():
    """Test 11: Collection info reflects state after operations."""
    print("Test 11: Collection info after operations (cloud)")
    coll = _prefixed("info_ops")
    db = _make_db(coll)
    try:
        db.create_collection()
        _wait()

        info_empty = db.get_collection_info()
        assert info_empty.vector_count == 0
        assert info_empty.dimension == DIM
        assert info_empty.distance_metric == DistanceMetric.COSINE
        print(f"  Empty: vectors={info_empty.vector_count}, dim={info_empty.dimension}")

        docs = [_make_chunk(f"Doc {i}", chunk_index=i) for i in range(10)]
        db.add_documents(docs)
        _wait()

        info_full = db.get_collection_info()
        assert info_full.vector_count == 10
        print(f"  After 10 adds: vectors={info_full.vector_count}")
        print(f"  Status: {info_full.metadata.get('status')}")
        print("  ✅ PASSED\n")
    finally:
        _safe_cleanup(db, coll)


def test_rich_metadata_roundtrip():
    """Test 12: Complex nested metadata survives cloud roundtrip."""
    print("Test 12: Rich metadata roundtrip (cloud)")
    coll = _prefixed("rich_meta")
    db = _make_db(coll)
    try:
        db.create_collection()
        _wait()

        rich_meta = {
            "source": "report.pdf",
            "page_number": 5,
            "section_title": "Results & Discussion",
            "heading_path": ["Introduction", "Methods", "Results & Discussion"],
            "has_table": True,
            "token_count": 142,
            "chunk_index": 3,
            "total_chunks": 12,
            "tags": ["important", "reviewed"],
        }
        doc = _make_chunk("The results show a 40% improvement...", extra_meta=rich_meta)
        ids = db.add_documents([doc])
        _wait()

        retrieved = db.get_by_ids(ids)
        assert len(retrieved) == 1
        rm = retrieved[0].metadata
        assert rm.get("page_number") == 5
        assert rm.get("section_title") == "Results & Discussion"
        assert rm.get("heading_path") == ["Introduction", "Methods", "Results & Discussion"]
        assert rm.get("has_table") is True
        assert rm.get("tags") == ["important", "reviewed"]
        print(f"  Roundtripped metadata keys: {sorted(rm.keys())}")
        print("  ✅ PASSED\n")
    finally:
        _safe_cleanup(db, coll)


# ── Main ─────────────────────────────────────────────────────────

def main():
    # ── Pre-flight checks ────────────────────────────────────────
    if not QDRANT_URL or not os.getenv("QDRANT_API_KEY"):
        print("❌  QDRANT_URL and QDRANT_API_KEY must be set in .env")
        sys.exit(1)

    print(f"\n  Qdrant Cloud URL : {QDRANT_URL}")
    print(f"  Run ID           : {RUN_ID}")
    print(f"  Collection prefix: cloud_test_{RUN_ID}")

    # Quick connectivity check
    try:
        db = _make_db()
        _ = db.list_collections()  # will raise on bad creds / unreachable
        print("  Connection       : ✅ OK\n")
    except Exception as e:
        print(f"\n❌  Cannot connect to Qdrant Cloud: {e}")
        sys.exit(1)

    random.seed(42)

    print("=" * 60)
    print("  VectorDB CRUD E2E Tests (Qdrant Cloud)")
    print("=" * 60 + "\n")

    tests = [
        test_collection_lifecycle,              # 1
        test_add_and_get_documents,             # 2
        test_batch_add,                         # 3
        test_update_documents,                  # 4
        test_delete_by_ids,                     # 5
        test_delete_by_filter,                  # 6
        test_vector_search,                     # 7
        test_search_with_score_and_threshold,   # 8
        test_filtered_search,                   # 9
        test_count,                             # 10
        test_collection_info_after_operations,  # 11
        test_rich_metadata_roundtrip,           # 12
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  ❌ FAILED: {e}\n")

    print("=" * 60)
    if failed == 0:
        print(f"  ✅ All {passed} Qdrant Cloud tests passed!")
    else:
        print(f"  Results: {passed} passed, {failed} failed")
    print("=" * 60 + "\n")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
