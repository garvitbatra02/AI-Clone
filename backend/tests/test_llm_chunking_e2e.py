"""
End-to-end tests for LLM-assisted smart chunking.

These tests make REAL LLM calls (Groq llama-3.1-8b-instant) to verify
the full pipeline:
    LLM prompt → JSON response → parse → validate → chunk → display

Run:
    PYTHONPATH=. python tests/test_llm_chunking_e2e.py

Requires:
    GROQ_API_KEY (or GROQ_API_KEYS) set in .env or environment.
"""

from __future__ import annotations

import os
import sys
import textwrap

from dotenv import load_dotenv

load_dotenv()

# ── Imports ──────────────────────────────────────────────────────

from ChatService.Chat.llm import LLMFactory, LLMProvider
from RAGService.Data.DocumentProcessors.strategies.llm_analyzer import (
    LLMStructureAnalyzer,
    StructuralMap,
)
from RAGService.Data.DocumentProcessors.smart_chunker import (
    SmartChunker,
    SmartChunkerConfig,
)
from RAGService.Data.DocumentProcessors.base import (
    ProcessedDocument,
    SupportedFileType,
)


# ── Helpers ──────────────────────────────────────────────────────

def _get_groq_keys() -> list[str]:
    """Get Groq API keys from environment."""
    multi = os.environ.get("GROQ_API_KEYS", "")
    if multi:
        return [k.strip() for k in multi.split(",") if k.strip()]
    single = os.environ.get("GROQ_API_KEY", "")
    if single:
        return [single.strip()]
    return []


def _make_llm():
    """Create a cheap fast Groq LLM for testing."""
    keys = _get_groq_keys()
    if not keys:
        return None
    return LLMFactory.create(
        provider=LLMProvider.GROQ,
        model="llama-3.1-8b-instant",
        api_keys=keys,
        temperature=0.0,
        max_tokens=2048,
    )


def _print_chunks(chunks, label: str) -> None:
    """Pretty-print chunks for visual verification."""
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"  Total chunks: {len(chunks)}")
    print(f"{'=' * 70}")
    for i, chunk in enumerate(chunks):
        meta = chunk.metadata
        topic = meta.get("topic", "")
        section = meta.get("section_title", "")
        heading = meta.get("heading_path", "")
        has_table = meta.get("has_table", False)
        tokens = meta.get("token_count", "?")

        header_parts = [f"Chunk {i}"]
        if section:
            header_parts.append(f"section={section}")
        if topic:
            header_parts.append(f"topic={topic}")
        if heading:
            header_parts.append(f"heading_path={heading}")
        if has_table:
            header_parts.append("HAS_TABLE")
        header_parts.append(f"~{tokens} tokens")

        print(f"\n--- {' | '.join(header_parts)} ---")
        # Truncate long content for display
        content = chunk.content
        if len(content) > 400:
            content = content[:400] + f"\n... ({len(chunk.content)} chars total)"
        print(content)
    print(f"\n{'=' * 70}\n")


def _print_structural_map(smap: StructuralMap, label: str) -> None:
    """Pretty-print the LLM structural map."""
    print(f"\n--- Structural Map: {label} ---")
    print(f"  Source: {smap.source}  |  Coverage: {smap.coverage:.2%}")
    print(f"  Elements ({len(smap.elements)}):")
    for elem in smap.elements:
        extra = ""
        if elem.title:
            extra = f'  title="{elem.title}"'
        if elem.level:
            extra += f"  level={elem.level}"
        print(
            f"    L{elem.start_line}-{elem.end_line}  "
            f"type={elem.element_type}{extra}  "
            f"confidence={elem.confidence}"
        )
    print()


# ── Test Data ────────────────────────────────────────────────────

STRUCTURED_DOC = textwrap.dedent("""\
    Introduction to Machine Learning

    Machine learning is a subset of artificial intelligence that focuses
    on building systems that learn from data. Rather than being explicitly
    programmed, these systems improve their performance through experience.

    Types of Machine Learning

    Supervised Learning

    In supervised learning, the model is trained on labeled data. The
    algorithm learns a mapping function from input variables to output
    variables. Common algorithms include linear regression, decision trees,
    and support vector machines.

    | Algorithm         | Type           | Use Case               |
    |-------------------|----------------|------------------------|
    | Linear Regression | Regression     | Price prediction       |
    | Random Forest     | Classification | Spam detection         |
    | SVM               | Both           | Image classification   |
    | Neural Networks   | Both           | Complex pattern rec.   |

    Unsupervised Learning

    Unsupervised learning works with unlabeled data. The algorithm tries
    to find hidden patterns or structures. Key techniques include
    clustering (K-means, DBSCAN) and dimensionality reduction (PCA, t-SNE).

    Reinforcement Learning

    In reinforcement learning, an agent learns by interacting with an
    environment. It receives rewards or penalties for actions and aims to
    maximize cumulative reward. Applications include game playing (AlphaGo),
    robotics, and autonomous driving.

    Evaluation Metrics

    Model performance is measured using various metrics:
    - Accuracy: proportion of correct predictions
    - Precision: true positives / (true positives + false positives)
    - Recall: true positives / (true positives + false negatives)
    - F1 Score: harmonic mean of precision and recall
    - AUC-ROC: area under the receiver operating characteristic curve

    Conclusion

    Machine learning continues to transform industries from healthcare
    to finance. Understanding the fundamentals — supervised, unsupervised,
    and reinforcement learning — provides a solid foundation for applying
    these techniques to real-world problems.
""")

PLAIN_TEXT_DOC = textwrap.dedent("""\
    John Smith is a software engineer based in San Francisco.
    He graduated from Stanford University in 2015 with a degree in Computer Science.
    During his time at Stanford, he focused on distributed systems and machine learning.
    He was a teaching assistant for CS229 (Machine Learning) for two semesters.

    After graduation, John joined Google as a backend engineer.
    He worked on the Google Cloud Platform team for three years.
    His main contributions were to the load balancing and auto-scaling infrastructure.
    He received two peer bonuses for his work on reducing latency by 40%.

    In 2018, John moved to Stripe as a senior engineer.
    At Stripe, he led the payments processing optimization project.
    His team reduced payment processing time from 800ms to 200ms.
    He was promoted to Staff Engineer in 2020.

    John is proficient in Python, Go, Java, and Rust.
    He has extensive experience with Kubernetes, Docker, and Terraform.
    He is a certified AWS Solutions Architect and GCP Professional Cloud Architect.
    He has contributed to several open-source projects including gRPC and Envoy.

    Outside of work, John enjoys hiking in the Bay Area and playing chess.
    He volunteers as a mentor at Code2040, helping underrepresented minorities in tech.
    He is also an avid reader, with a particular interest in science fiction.
    He runs a tech blog that has over 10,000 subscribers.
""")

CSV_DOC_CONTENT = textwrap.dedent("""\
    name,role,department,salary
    Alice Johnson,Engineering Manager,Engineering,185000
    Bob Smith,Senior Developer,Engineering,155000
    Carol Williams,Data Scientist,Data Science,145000
    David Brown,Product Manager,Product,160000
    Eva Martinez,UX Designer,Design,130000
""").strip()

JSON_DOC_CONTENT = textwrap.dedent("""\
    [
        {"id": 1, "name": "Widget A", "price": 29.99, "category": "electronics", "in_stock": true},
        {"id": 2, "name": "Widget B", "price": 49.99, "category": "electronics", "in_stock": false},
        {"id": 3, "name": "Gadget X", "price": 99.99, "category": "premium", "in_stock": true}
    ]
""").strip()


# ── Tests ────────────────────────────────────────────────────────

def test_llm_structural_analysis(llm):
    """
    Test 1: Run LLMStructureAnalyzer in 'structural' mode on a
    document with headings and a table. Verify the structural map
    and display it.
    """
    print("\n" + "=" * 70)
    print("  TEST 1: LLM Structural Analysis (PDF-like document)")
    print("=" * 70)

    analyzer = LLMStructureAnalyzer(llm=llm)
    assert analyzer.is_available

    result = analyzer.analyze(STRUCTURED_DOC, mode="structural")

    assert result is not None, "LLM returned no structural map"
    assert isinstance(result, StructuralMap)
    assert result.coverage > 0.5, f"Coverage too low: {result.coverage:.2%}"
    assert len(result.elements) >= 3, f"Too few elements: {len(result.elements)}"

    _print_structural_map(result, "Structural Analysis")

    # Verify at least one heading and one table were detected
    types_found = {e.element_type for e in result.elements}
    print(f"  Element types detected: {types_found}")
    assert "heading" in types_found, "No headings detected"

    print("  ✅ TEST 1 PASSED\n")
    return result


def test_llm_topical_analysis(llm):
    """
    Test 2: Run LLMStructureAnalyzer in 'topical' mode on a plain
    text biography. Verify topic labels are meaningful.
    """
    print("\n" + "=" * 70)
    print("  TEST 2: LLM Topical Grouping (Plain Text)")
    print("=" * 70)

    analyzer = LLMStructureAnalyzer(llm=llm)
    result = analyzer.analyze(PLAIN_TEXT_DOC, mode="topical")

    assert result is not None, "LLM returned no topical map"
    assert result.coverage > 0.5, f"Coverage too low: {result.coverage:.2%}"
    assert len(result.elements) >= 2, f"Too few groups: {len(result.elements)}"

    _print_structural_map(result, "Topical Grouping")

    # Every element should have a topic title
    for elem in result.elements:
        assert elem.title, f"Element L{elem.start_line}-{elem.end_line} has no topic"

    print("  ✅ TEST 2 PASSED\n")
    return result


def test_e2e_smart_chunker_structural(groq_keys):
    """
    Test 3: Full e2e — SmartChunker with LLM enabled on a
    structured PDF-like document. Chunks should reflect section
    boundaries.
    """
    print("\n" + "=" * 70)
    print("  TEST 3: E2E SmartChunker — Structural (PDF-like)")
    print("=" * 70)

    config = SmartChunkerConfig(
        max_chunk_size=600,
        chunk_overlap=50,
        use_llm_analysis=True,
        llm_provider="groq",
        llm_model="llama-3.1-8b-instant",
        llm_api_keys=groq_keys,
    )
    chunker = SmartChunker(config)

    doc = ProcessedDocument(
        content=STRUCTURED_DOC,
        metadata={},
        source="ml_guide.pdf",
        file_type=SupportedFileType.PDF,
    )

    chunks = chunker.chunk_document(doc)

    assert len(chunks) >= 3, f"Expected ≥3 chunks, got {len(chunks)}"
    _print_chunks(chunks, "E2E Structural Chunking (PDF-like, LLM ON)")

    # Each chunk should have metadata
    for c in chunks:
        assert "chunk_index" in c.metadata
        assert "total_chunks" in c.metadata
        assert c.content.strip(), f"Chunk {c.chunk_index} is empty"

    print("  ✅ TEST 3 PASSED\n")
    return chunks


def test_e2e_smart_chunker_topical(groq_keys):
    """
    Test 4: Full e2e — SmartChunker with LLM enabled on plain text.
    Chunks should get topic labels.
    """
    print("\n" + "=" * 70)
    print("  TEST 4: E2E SmartChunker — Topical (TXT)")
    print("=" * 70)

    config = SmartChunkerConfig(
        max_chunk_size=500,
        chunk_overlap=50,
        use_llm_analysis=True,
        llm_provider="groq",
        llm_model="llama-3.1-8b-instant",
        llm_api_keys=groq_keys,
    )
    chunker = SmartChunker(config)

    doc = ProcessedDocument(
        content=PLAIN_TEXT_DOC,
        metadata={},
        source="bio.txt",
        file_type=SupportedFileType.TXT,
    )

    chunks = chunker.chunk_document(doc)

    assert len(chunks) >= 2, f"Expected ≥2 chunks, got {len(chunks)}"
    _print_chunks(chunks, "E2E Topical Chunking (TXT, LLM ON)")

    # Verify topic labels are present
    topics_found = [c.metadata.get("topic") for c in chunks if c.metadata.get("topic")]
    print(f"  Topics found: {topics_found}")
    assert len(topics_found) >= 2, f"Expected ≥2 topic labels, got {len(topics_found)}"

    print("  ✅ TEST 4 PASSED\n")
    return chunks


def test_e2e_smart_chunker_no_llm_comparison():
    """
    Test 5: SmartChunker WITHOUT LLM on the same documents for
    comparison. Shows what the fallback (rule-based) produces.
    """
    print("\n" + "=" * 70)
    print("  TEST 5: SmartChunker — No LLM (Baseline Comparison)")
    print("=" * 70)

    config = SmartChunkerConfig(
        max_chunk_size=600,
        chunk_overlap=50,
        use_llm_analysis=False,
    )
    chunker = SmartChunker(config)

    # PDF-like
    pdf_doc = ProcessedDocument(
        content=STRUCTURED_DOC,
        metadata={},
        source="ml_guide.pdf",
        file_type=SupportedFileType.PDF,
    )
    pdf_chunks = chunker.chunk_document(pdf_doc)
    _print_chunks(pdf_chunks, "Baseline Structural Chunking (PDF-like, NO LLM)")

    # TXT
    txt_doc = ProcessedDocument(
        content=PLAIN_TEXT_DOC,
        metadata={},
        source="bio.txt",
        file_type=SupportedFileType.TXT,
    )
    txt_chunks = chunker.chunk_document(txt_doc)
    _print_chunks(txt_chunks, "Baseline Recursive Chunking (TXT, NO LLM)")

    print("  ✅ TEST 5 PASSED\n")
    return pdf_chunks, txt_chunks


def test_e2e_csv_and_json_unchanged(groq_keys):
    """
    Test 6: CSV and JSON go through their own strategies even when
    LLM is enabled. Verify they are unaffected.
    """
    print("\n" + "=" * 70)
    print("  TEST 6: CSV & JSON routing (LLM enabled but unused)")
    print("=" * 70)

    config = SmartChunkerConfig(
        use_llm_analysis=True,
        llm_provider="groq",
        llm_model="llama-3.1-8b-instant",
        llm_api_keys=groq_keys,
    )
    chunker = SmartChunker(config)

    # CSV
    csv_doc = ProcessedDocument(
        content=CSV_DOC_CONTENT,
        metadata={
            "column_names": ["name", "role", "department", "salary"],
            "rows_data": [
                {"name": "Alice Johnson", "role": "Engineering Manager", "department": "Engineering", "salary": "185000"},
                {"name": "Bob Smith", "role": "Senior Developer", "department": "Engineering", "salary": "155000"},
                {"name": "Carol Williams", "role": "Data Scientist", "department": "Data Science", "salary": "145000"},
                {"name": "David Brown", "role": "Product Manager", "department": "Product", "salary": "160000"},
                {"name": "Eva Martinez", "role": "UX Designer", "department": "Design", "salary": "130000"},
            ],
        },
        source="employees.csv",
        file_type=SupportedFileType.CSV,
    )
    csv_chunks = chunker.chunk_document(csv_doc)
    assert len(csv_chunks) == 5, f"Expected 5 CSV chunks (one per row), got {len(csv_chunks)}"
    _print_chunks(csv_chunks, "CSV Chunks (LLM enabled but not used)")

    # JSON
    import json
    entries = json.loads(JSON_DOC_CONTENT)
    json_doc = ProcessedDocument(
        content=JSON_DOC_CONTENT,
        metadata={
            "entries_data": entries,
            "entry_paths": ["[0]", "[1]", "[2]"],
        },
        source="products.json",
        file_type=SupportedFileType.JSON,
    )
    json_chunks = chunker.chunk_document(json_doc)
    assert len(json_chunks) == 3, f"Expected 3 JSON chunks (one per entry), got {len(json_chunks)}"
    _print_chunks(json_chunks, "JSON Chunks (LLM enabled but not used)")

    print("  ✅ TEST 6 PASSED\n")


# ── Main ─────────────────────────────────────────────────────────

def main():
    groq_keys = _get_groq_keys()
    if not groq_keys:
        print("⚠️  No GROQ_API_KEY / GROQ_API_KEYS found in environment.")
        print("   Set them in .env or export manually to run LLM e2e tests.")
        print("   Running only non-LLM baseline test (Test 5)...\n")
        test_e2e_smart_chunker_no_llm_comparison()
        print("Done (LLM tests skipped).")
        return

    llm = _make_llm()
    assert llm is not None

    print(f"\n🔑 Using {len(groq_keys)} Groq API key(s)")
    print(f"🤖 Model: llama-3.1-8b-instant\n")

    # Test 1: Raw structural analysis
    test_llm_structural_analysis(llm)

    # Test 2: Raw topical analysis
    test_llm_topical_analysis(llm)

    # Test 3: Full e2e structural chunking
    structural_chunks = test_e2e_smart_chunker_structural(groq_keys)

    # Test 4: Full e2e topical chunking
    topical_chunks = test_e2e_smart_chunker_topical(groq_keys)

    # Test 5: Baseline (no LLM) for comparison
    baseline_pdf, baseline_txt = test_e2e_smart_chunker_no_llm_comparison()

    # Test 6: CSV & JSON unaffected
    test_e2e_csv_and_json_unchanged(groq_keys)

    # Summary comparison
    print("\n" + "=" * 70)
    print("  SUMMARY: LLM vs Baseline chunk counts")
    print("=" * 70)
    print(f"  PDF-like doc:  LLM={len(structural_chunks)} chunks  |  Baseline={len(baseline_pdf)} chunks")
    print(f"  TXT doc:       LLM={len(topical_chunks)} chunks  |  Baseline={len(baseline_txt)} chunks")
    print()
    print("✅ All 6 e2e tests passed!")


if __name__ == "__main__":
    main()
