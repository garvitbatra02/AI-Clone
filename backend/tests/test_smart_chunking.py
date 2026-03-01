"""Quick smoke test for LLMStructureAnalyzer, SmartChunker, and the rewired AssetUploadService."""

from RAGService.Data.DocumentProcessors.strategies.llm_analyzer import (
    LLMStructureAnalyzer,
    StructuralElement,
    StructuralMap,
)
from RAGService.Data.DocumentProcessors.smart_chunker import (
    SmartChunker,
    SmartChunkerConfig,
)
from RAGService.Data.DocumentProcessors.base import ProcessedDocument, SupportedFileType
from RAGService.Data.services.asset_upload_service import AssetUploadService, AssetUploadConfig


def test_analyzer_disabled():
    analyzer = LLMStructureAnalyzer(llm=None)
    assert not analyzer.is_available
    assert analyzer.analyze("test") is None
    print("1. Disabled analyzer returns None: OK")


def test_parse_markdown_fence():
    a = LLMStructureAnalyzer.__new__(LLMStructureAnalyzer)
    raw = '```json\n[{"start_line": 1, "end_line": 5, "type": "heading", "level": 1, "title": "Intro"}, {"start_line": 6, "end_line": 20, "type": "body", "level": null, "title": null}]\n```'
    elements = a._parse_response(raw)
    assert len(elements) == 2
    assert elements[0]["type"] == "heading"
    print("2. Parse from markdown fence: OK")


def test_parse_direct_json():
    a = LLMStructureAnalyzer.__new__(LLMStructureAnalyzer)
    raw = '[{"start_line": 1, "end_line": 3, "type": "body"}]'
    elements = a._parse_response(raw)
    assert len(elements) == 1
    print("3. Parse direct JSON: OK")


def test_parse_bracket_extraction():
    a = LLMStructureAnalyzer.__new__(LLMStructureAnalyzer)
    raw = 'Here is the result: [{"start_line": 1, "end_line": 2, "type": "table"}] done.'
    elements = a._parse_response(raw)
    assert len(elements) == 1
    assert elements[0]["type"] == "table"
    print("4. Parse bracket extraction: OK")


def test_validation():
    a = LLMStructureAnalyzer.__new__(LLMStructureAnalyzer)
    raw_elements = [
        {"start_line": 1, "end_line": 1, "type": "heading", "level": 1, "title": "Title"},
        {"start_line": 2, "end_line": 5, "type": "body", "level": None, "title": None},
    ]
    lines = ["# Title", "Some body text.", "More text.", "| col1 | col2 |", "| a | b |"]
    validated = a._validate_elements(raw_elements, lines)
    assert len(validated) == 2
    assert validated[0].element_type == "heading"
    assert validated[0].title == "Title"
    print("5. Validation cross-check: OK")


def test_smart_chunker_csv():
    config = SmartChunkerConfig(max_chunk_size=500)
    chunker = SmartChunker(config)

    doc = ProcessedDocument(
        content="name,age\nAlice,30\nBob,25\nCharlie,35",
        metadata={
            "column_names": ["name", "age"],
            "rows_data": [
                {"name": "Alice", "age": "30"},
                {"name": "Bob", "age": "25"},
                {"name": "Charlie", "age": "35"},
            ],
        },
        source="data.csv",
        file_type=SupportedFileType.CSV,
    )
    chunks = chunker.chunk_document(doc, SupportedFileType.CSV)
    assert len(chunks) == 3, f"Expected 3 chunks, got {len(chunks)}"
    assert "Alice" in chunks[0].content
    print("6. SmartChunker CSV routing: OK")


def test_smart_chunker_json():
    config = SmartChunkerConfig()
    chunker = SmartChunker(config)

    doc = ProcessedDocument(
        content='[{"id": 1, "name": "A"}, {"id": 2, "name": "B"}]',
        metadata={
            "entries_data": [{"id": 1, "name": "A"}, {"id": 2, "name": "B"}],
            "entry_paths": ["[0]", "[1]"],
        },
        source="data.json",
        file_type=SupportedFileType.JSON,
    )
    chunks = chunker.chunk_document(doc, SupportedFileType.JSON)
    assert len(chunks) == 2, f"Expected 2 chunks, got {len(chunks)}"
    print("7. SmartChunker JSON routing: OK")


def test_smart_chunker_txt():
    config = SmartChunkerConfig(max_chunk_size=200, chunk_overlap=0)
    chunker = SmartChunker(config)

    doc = ProcessedDocument(
        content="First paragraph about AI.\n\nSecond paragraph about ML.\n\nThird paragraph about DL.",
        metadata={},
        source="notes.txt",
        file_type=SupportedFileType.TXT,
    )
    chunks = chunker.chunk_document(doc, SupportedFileType.TXT)
    assert len(chunks) >= 1
    print(f"8. SmartChunker TXT routing: OK ({len(chunks)} chunks)")


def test_asset_upload_config():
    config = AssetUploadConfig(use_smart_chunker=True, use_llm_analysis=False)
    assert config.use_smart_chunker is True
    assert config.use_llm_analysis is False
    print("9. AssetUploadConfig new fields: OK")


if __name__ == "__main__":
    test_analyzer_disabled()
    test_parse_markdown_fence()
    test_parse_direct_json()
    test_parse_bracket_extraction()
    test_validation()
    test_smart_chunker_csv()
    test_smart_chunker_json()
    test_smart_chunker_txt()
    test_asset_upload_config()
    print("\n✅ All smoke tests passed!")
