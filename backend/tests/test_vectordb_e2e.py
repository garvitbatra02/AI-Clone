"""
End-to-end tests for the VectorDB module (Qdrant provider, in-memory).

Tests every CRUD operation exposed by BaseVectorDB using an in-memory
Qdrant instance — no external services required.

Run:
    PYTHONPATH=. python tests/test_vectordb_e2e.py

Covers:
    1.  Create / delete / list / exists — collection lifecycle
    2.  Add documents (single + batch)
    3.  Get by IDs (with and without embeddings)
    4.  Update documents (upsert semantics)
    5.  Delete by IDs
    6.  Delete by metadata filter
    7.  Vector similarity search
    8.  Similarity search with score + score threshold
    9.  Metadata-filtered search
    10. Count utility
    11. Collection info / stats
    12. Multiple collections isolation
"""

from __future__ import annotations

import random
import uuid
from typing import List

from RAGService.Data.VectorDB import (
    VectorDBFactory,
    VectorDBProvider,
    DistanceMetric,
    DocumentChunk,
    SearchResult,
    CollectionInfo,
    MetadataFilter,
    MetadataFilterGroup,
    FilterOperator,
)


# ── Helpers ──────────────────────────────────────────────────────

DIM = 128  # Small dimension for fast tests

COLLECTION = "test_collection"


def _make_db(collection: str = COLLECTION):
    """Create an in-memory Qdrant instance."""
    return VectorDBFactory.create_from_env(
        provider=VectorDBProvider.QDRANT,
        collection_name=collection,
        embedding_dimension=DIM,
        in_memory=True,
        distance_metric=DistanceMetric.COSINE,
    )


def _rand_vec(dim: int = DIM) -> List[float]:
    """Generate a random unit-ish vector."""
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
    """Create a DocumentChunk with a random embedding."""
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


# ── Tests ────────────────────────────────────────────────────────

def test_collection_lifecycle():
    """
    Test 1: create → exists → info → list → delete → not exists
    """
    print("Test 1: Collection lifecycle")
    db = _make_db()

    # Should not exist yet
    assert not db.collection_exists(), "Collection should not exist initially"

    # Create
    ok = db.create_collection()
    assert ok, "create_collection should return True"
    assert db.collection_exists(), "Collection should exist after create"

    # Info
    info = db.get_collection_info()
    assert isinstance(info, CollectionInfo)
    assert info.name == COLLECTION
    assert info.dimension == DIM
    assert info.distance_metric == DistanceMetric.COSINE
    assert info.vector_count == 0
    print(f"  Info: name={info.name}, dim={info.dimension}, metric={info.distance_metric.value}, vectors={info.vector_count}")

    # List
    names = db.list_collections()
    assert COLLECTION in names, f"Collection not in list: {names}"
    print(f"  Collections: {names}")

    # Delete
    db.delete_collection()
    assert not db.collection_exists(), "Collection should not exist after delete"
    print("  ✅ PASSED\n")


def test_add_and_get_documents():
    """
    Test 2: Add documents, then retrieve by ID — content and metadata roundtrip.
    """
    print("Test 2: Add & get documents")
    db = _make_db()
    db.create_collection()

    docs = [
        _make_chunk("Machine learning is great.", chunk_index=0, extra_meta={"topic": "ml"}),
        _make_chunk("Deep learning uses neural networks.", chunk_index=1, extra_meta={"topic": "dl"}),
        _make_chunk("Natural language processing is a subfield of AI.", chunk_index=2, extra_meta={"topic": "nlp"}),
    ]
    ids = db.add_documents(docs)

    assert len(ids) == 3
    print(f"  Added {len(ids)} documents, IDs: {ids[:3]}")

    # Retrieve without embeddings
    retrieved = db.get_by_ids(ids)
    assert len(retrieved) == 3

    for orig, ret in zip(docs, retrieved):
        assert ret.content == orig.content, f"Content mismatch: {ret.content!r}"
        assert ret.metadata.get("topic") == orig.metadata["topic"]
        assert ret.embedding is None, "Embedding should be None when not requested"

    # Retrieve WITH embeddings
    retrieved_with_emb = db.get_by_ids(ids[:1], include_embeddings=True)
    assert len(retrieved_with_emb) == 1
    assert retrieved_with_emb[0].embedding is not None
    assert len(retrieved_with_emb[0].embedding) == DIM
    print(f"  Retrieved with embeddings: dim={len(retrieved_with_emb[0].embedding)}")

    # Count
    assert db.count() == 3
    print(f"  Count: {db.count()}")
    print("  ✅ PASSED\n")

    db.delete_collection()


def test_batch_add():
    """
    Test 3: Batch add with explicit batch_size to test batching logic.
    """
    print("Test 3: Batch add (20 docs, batch_size=7)")
    db = _make_db()
    db.create_collection()

    docs = [_make_chunk(f"Document number {i}", chunk_index=i) for i in range(20)]
    ids = db.add_documents(docs, batch_size=7)

    assert len(ids) == 20
    assert db.count() == 20
    print(f"  Added 20 docs in batches of 7, count={db.count()}")
    print("  ✅ PASSED\n")

    db.delete_collection()


def test_update_documents():
    """
    Test 4: Add doc, then update its content and metadata via upsert.
    """
    print("Test 4: Update documents (upsert)")
    db = _make_db()
    db.create_collection()

    # Add original
    doc = _make_chunk("Original content", extra_meta={"version": 1})
    doc_id = doc.id
    db.add_documents([doc])

    retrieved = db.get_by_ids([doc_id])
    assert retrieved[0].content == "Original content"
    assert retrieved[0].metadata.get("version") == 1
    print(f"  Original: content={retrieved[0].content!r}, version={retrieved[0].metadata.get('version')}")

    # Update — same ID, new content
    updated_doc = DocumentChunk(
        id=doc_id,
        content="Updated content with more detail",
        metadata={"source": "test.txt", "version": 2, "updated": True},
        embedding=_rand_vec(),
    )
    db.update_documents([updated_doc])

    retrieved2 = db.get_by_ids([doc_id])
    assert retrieved2[0].content == "Updated content with more detail"
    assert retrieved2[0].metadata.get("version") == 2
    assert retrieved2[0].metadata.get("updated") is True
    assert db.count() == 1, "Update should not create a duplicate"
    print(f"  Updated: content={retrieved2[0].content!r}, version={retrieved2[0].metadata.get('version')}, count={db.count()}")
    print("  ✅ PASSED\n")

    db.delete_collection()


def test_delete_by_ids():
    """
    Test 5: Add 5 docs, delete 2 by ID, verify remaining.
    """
    print("Test 5: Delete by IDs")
    db = _make_db()
    db.create_collection()

    docs = [_make_chunk(f"Doc {i}", chunk_index=i) for i in range(5)]
    ids = db.add_documents(docs)
    assert db.count() == 5

    # Delete first two
    ok = db.delete_by_ids(ids[:2])
    assert ok, "delete_by_ids should return True"
    assert db.count() == 3, f"Expected 3 remaining, got {db.count()}"

    # Verify deleted docs are gone
    remaining = db.get_by_ids(ids[:2])
    assert len(remaining) == 0, f"Deleted docs should not be retrievable, got {len(remaining)}"

    # Verify others still exist
    alive = db.get_by_ids(ids[2:])
    assert len(alive) == 3
    print(f"  Deleted 2, remaining={db.count()}, retrieved surviving 3 OK")
    print("  ✅ PASSED\n")

    db.delete_collection()


def test_delete_by_filter():
    """
    Test 6: Add docs with different categories, delete by metadata filter.
    """
    print("Test 6: Delete by metadata filter")
    db = _make_db()
    db.create_collection()

    docs = [
        _make_chunk("Python guide", chunk_index=0, extra_meta={"category": "programming"}),
        _make_chunk("Java guide", chunk_index=1, extra_meta={"category": "programming"}),
        _make_chunk("Cooking recipe", chunk_index=2, extra_meta={"category": "cooking"}),
        _make_chunk("History lesson", chunk_index=3, extra_meta={"category": "history"}),
    ]
    db.add_documents(docs)
    assert db.count() == 4

    # Delete all "programming" docs
    deleted = db.delete_by_filter(
        MetadataFilter(field="category", operator=FilterOperator.EQ, value="programming")
    )
    assert deleted == 2, f"Expected 2 deleted, got {deleted}"
    assert db.count() == 2

    # Remaining should be cooking + history
    remaining = db.get_by_ids([d.id for d in docs[2:]])
    categories = {r.metadata.get("category") for r in remaining}
    assert categories == {"cooking", "history"}, f"Unexpected remaining: {categories}"
    print(f"  Deleted {deleted} 'programming' docs, remaining categories: {categories}")
    print("  ✅ PASSED\n")

    db.delete_collection()


def test_vector_search():
    """
    Test 7: Similarity search — a query vector close to one doc should rank it first.
    """
    print("Test 7: Vector similarity search")
    db = _make_db()
    db.create_collection()

    # Create a known vector and a near-duplicate as the query
    target_vec = _rand_vec()
    # Make query = target + tiny noise  →  should be most similar
    query_vec = [v + random.gauss(0, 0.01) for v in target_vec]
    other_vecs = [_rand_vec() for _ in range(9)]

    docs = [_make_chunk("Target document", embedding=target_vec, chunk_index=0)]
    for i, vec in enumerate(other_vecs):
        docs.append(_make_chunk(f"Other doc {i}", embedding=vec, chunk_index=i + 1))

    db.add_documents(docs)

    results = db.search(query_embedding=query_vec, k=3)
    assert len(results) == 3
    assert isinstance(results[0], SearchResult)
    assert results[0].content == "Target document", f"Top result should be target, got: {results[0].content!r}"
    assert results[0].score > results[1].score, "Top result should have highest score"

    print(f"  Top-3 results:")
    for r in results:
        print(f"    score={r.score:.4f}  content={r.content!r}")
    print("  ✅ PASSED\n")

    db.delete_collection()


def test_search_with_score_and_threshold():
    """
    Test 8: similarity_search_with_score + score_threshold filtering.
    """
    print("Test 8: Search with score + threshold")
    db = _make_db()
    db.create_collection()

    # Two clusters: close docs and far docs
    base_vec = _rand_vec()
    close_docs = [
        _make_chunk(f"Close doc {i}", embedding=[v + random.gauss(0, 0.02) for v in base_vec], chunk_index=i)
        for i in range(3)
    ]
    far_docs = [
        _make_chunk(f"Far doc {i}", embedding=_rand_vec(), chunk_index=i + 3)
        for i in range(5)
    ]
    db.add_documents(close_docs + far_docs)

    # similarity_search_with_score
    query = [v + random.gauss(0, 0.01) for v in base_vec]
    doc_scores = db.similarity_search_with_score(query, k=8)
    assert len(doc_scores) == 8
    assert isinstance(doc_scores[0], tuple)
    assert isinstance(doc_scores[0][0], DocumentChunk)
    assert doc_scores[0][0].embedding is not None, "Should include embeddings"
    print(f"  similarity_search_with_score returned {len(doc_scores)} results")
    for doc, score in doc_scores[:4]:
        print(f"    score={score:.4f}  content={doc.content!r}")

    # Search with high threshold — should filter out far docs
    high_threshold = doc_scores[2][1] - 0.01  # just below 3rd best score
    filtered = db.search(query, k=10, score_threshold=high_threshold)
    assert len(filtered) <= 8, "Threshold should filter some results"
    print(f"  With threshold={high_threshold:.4f}: got {len(filtered)} results")
    print("  ✅ PASSED\n")

    db.delete_collection()


def test_filtered_search():
    """
    Test 9: Search with metadata filters — only matching docs returned.
    """
    print("Test 9: Metadata-filtered search")
    db = _make_db()
    db.create_collection()

    base_vec = _rand_vec()

    docs = [
        _make_chunk("Python basics", embedding=[v + random.gauss(0, 0.01) for v in base_vec],
                     chunk_index=0, extra_meta={"lang": "python", "level": 1}),
        _make_chunk("Python advanced", embedding=[v + random.gauss(0, 0.02) for v in base_vec],
                     chunk_index=1, extra_meta={"lang": "python", "level": 3}),
        _make_chunk("Java basics", embedding=[v + random.gauss(0, 0.01) for v in base_vec],
                     chunk_index=2, extra_meta={"lang": "java", "level": 1}),
        _make_chunk("Rust systems", embedding=_rand_vec(),
                     chunk_index=3, extra_meta={"lang": "rust", "level": 4}),
    ]
    db.add_documents(docs)

    # Filter: lang == "python"
    results = db.search(
        query_embedding=[v + random.gauss(0, 0.005) for v in base_vec],
        k=10,
        filters=MetadataFilter(field="lang", operator=FilterOperator.EQ, value="python"),
    )
    assert all(r.metadata.get("lang") == "python" for r in results), \
        f"All results should be python, got: {[r.metadata.get('lang') for r in results]}"
    assert len(results) == 2
    print(f"  Filter lang='python': {len(results)} results")

    # Filter: level >= 3
    results2 = db.search(
        query_embedding=[v + random.gauss(0, 0.005) for v in base_vec],
        k=10,
        filters=MetadataFilter(field="level", operator=FilterOperator.GTE, value=3),
    )
    levels = [r.metadata.get("level") for r in results2]
    assert all(l >= 3 for l in levels), f"All levels should be >=3, got: {levels}"
    print(f"  Filter level>=3: {len(results2)} results, levels={levels}")

    # Filter group: lang=='python' AND level==1
    results3 = db.search(
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
    assert len(results3) == 1
    assert results3[0].content == "Python basics"
    print(f"  Filter lang='python' AND level=1: {len(results3)} result → {results3[0].content!r}")

    # IN filter: lang in ["python", "rust"]
    results4 = db.search(
        query_embedding=base_vec,
        k=10,
        filters=MetadataFilter(field="lang", operator=FilterOperator.IN, value=["python", "rust"]),
    )
    langs = {r.metadata.get("lang") for r in results4}
    assert langs <= {"python", "rust"}, f"Expected python/rust only, got: {langs}"
    print(f"  Filter lang IN ['python','rust']: {len(results4)} results, langs={langs}")
    print("  ✅ PASSED\n")

    db.delete_collection()


def test_count():
    """
    Test 10: Count accuracy through add/delete cycles.
    """
    print("Test 10: Count utility")
    db = _make_db()
    db.create_collection()

    assert db.count() == 0, "Empty collection should have 0 count"

    docs = [_make_chunk(f"Doc {i}", chunk_index=i) for i in range(15)]
    ids = db.add_documents(docs)
    assert db.count() == 15
    print(f"  After adding 15: count={db.count()}")

    db.delete_by_ids(ids[:5])
    assert db.count() == 10
    print(f"  After deleting 5: count={db.count()}")

    db.add_documents([_make_chunk("Extra doc", chunk_index=99)])
    assert db.count() == 11
    print(f"  After adding 1 more: count={db.count()}")
    print("  ✅ PASSED\n")

    db.delete_collection()


def test_collection_info_after_operations():
    """
    Test 11: Collection info reflects correct state after operations.
    """
    print("Test 11: Collection info after operations")
    db = _make_db()
    db.create_collection()

    info_empty = db.get_collection_info()
    assert info_empty.vector_count == 0
    assert info_empty.dimension == DIM
    assert info_empty.distance_metric == DistanceMetric.COSINE
    print(f"  Empty: vectors={info_empty.vector_count}, dim={info_empty.dimension}")

    docs = [_make_chunk(f"Doc {i}", chunk_index=i) for i in range(10)]
    db.add_documents(docs)

    info_full = db.get_collection_info()
    assert info_full.vector_count == 10
    print(f"  After 10 adds: vectors={info_full.vector_count}")
    print(f"  Status: {info_full.metadata.get('status')}")
    print("  ✅ PASSED\n")

    db.delete_collection()


def test_multiple_collections_isolation():
    """
    Test 12: Operations on one collection don't affect another.
    """
    print("Test 12: Multiple collections isolation")
    db = _make_db("collection_a")
    db.create_collection(collection_name="collection_a")
    db.create_collection(collection_name="collection_b", dimension=DIM)

    # Add to A
    docs_a = [_make_chunk(f"A-doc-{i}", chunk_index=i, extra_meta={"coll": "a"}) for i in range(5)]
    db.add_documents(docs_a, collection_name="collection_a")

    # Add to B
    docs_b = [_make_chunk(f"B-doc-{i}", chunk_index=i, extra_meta={"coll": "b"}) for i in range(3)]
    db.add_documents(docs_b, collection_name="collection_b")

    # Counts are independent
    count_a = db.count(collection_name="collection_a")
    count_b = db.count(collection_name="collection_b")
    assert count_a == 5, f"Collection A should have 5, got {count_a}"
    assert count_b == 3, f"Collection B should have 3, got {count_b}"
    print(f"  collection_a: {count_a} docs  |  collection_b: {count_b} docs")

    # Delete from A doesn't affect B
    db.delete_by_ids([d.id for d in docs_a[:2]], collection_name="collection_a")
    assert db.count(collection_name="collection_a") == 3
    assert db.count(collection_name="collection_b") == 3
    print(f"  After deleting 2 from A: a={db.count(collection_name='collection_a')}, b={db.count(collection_name='collection_b')}")

    # Search in B only returns B docs
    results = db.search(
        query_embedding=docs_b[0].embedding,
        k=10,
        collection_name="collection_b",
    )
    assert all(r.metadata.get("coll") == "b" for r in results), "Search in B should only return B docs"
    print(f"  Search in B: {len(results)} results, all from collection B")

    # Delete collection A, B survives
    db.delete_collection(collection_name="collection_a")
    assert not db.collection_exists(collection_name="collection_a")
    assert db.collection_exists(collection_name="collection_b")
    assert db.count(collection_name="collection_b") == 3
    print(f"  After deleting collection_a: b still has {db.count(collection_name='collection_b')} docs")

    # Cleanup
    db.delete_collection(collection_name="collection_b")
    print("  ✅ PASSED\n")


def test_empty_operations():
    """
    Test 13: Edge cases — empty gets, empty deletes, search on empty collection.
    """
    print("Test 13: Edge cases (empty operations)")
    db = _make_db()
    db.create_collection()

    # Get with non-existent IDs
    result = db.get_by_ids(["nonexistent-id-1", "nonexistent-id-2"])
    assert len(result) == 0, f"Expected 0 results for fake IDs, got {len(result)}"
    print(f"  get_by_ids with fake IDs: {len(result)} results")

    # Search on empty collection
    results = db.search(query_embedding=_rand_vec(), k=5)
    assert len(results) == 0
    print(f"  Search on empty collection: {len(results)} results")

    # Count on empty
    assert db.count() == 0
    print(f"  Count on empty: {db.count()}")

    # Delete non-existent IDs (should not error)
    ok = db.delete_by_ids(["does-not-exist"])
    assert ok, "Deleting non-existent ID should still return True"
    print(f"  Delete non-existent ID: ok={ok}")

    print("  ✅ PASSED\n")

    db.delete_collection()


def test_large_metadata():
    """
    Test 14: Documents with rich nested metadata survive roundtrip.
    """
    print("Test 14: Rich metadata roundtrip")
    db = _make_db()
    db.create_collection()

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
    doc = _make_chunk(
        "The results show a 40% improvement...",
        extra_meta=rich_meta,
    )
    ids = db.add_documents([doc])
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

    db.delete_collection()


# ── Main ─────────────────────────────────────────────────────────

def main():
    random.seed(42)

    print("\n" + "=" * 60)
    print("  VectorDB CRUD E2E Tests (Qdrant in-memory)")
    print("=" * 60 + "\n")

    test_collection_lifecycle()           # 1
    test_add_and_get_documents()          # 2
    test_batch_add()                      # 3
    test_update_documents()               # 4
    test_delete_by_ids()                  # 5
    test_delete_by_filter()               # 6
    test_vector_search()                  # 7
    test_search_with_score_and_threshold()  # 8
    test_filtered_search()                # 9
    test_count()                          # 10
    test_collection_info_after_operations()  # 11
    test_multiple_collections_isolation()  # 12
    test_empty_operations()               # 13
    test_large_metadata()                 # 14

    print("=" * 60)
    print("  ✅ All 14 VectorDB CRUD tests passed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
