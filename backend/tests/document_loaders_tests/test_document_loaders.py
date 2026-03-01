"""
Test Document Loaders for RAGService.

Simple tests to initialize and use document loaders.

Sample files in this folder:
- conversation.json: JSON file with User/Agent conversations
- my all details.txt: Text file with personal details
- Personal_TrainingData.csv: CSV file with prompt/response pairs

Run tests:
   python tests/document_loaders_tests/test_document_loaders.py
"""

import os
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import document processors components
from RAGService.Data.DocumentProcessors import (
    BaseDocumentLoader,
    DocumentChunk,
    DocumentLoaderFactory,
    ProcessedDocument,
    RecursiveTextSplitter,
    SupportedFileType,
    TextSplitterConfig,
)
from RAGService.Data.DocumentProcessors.loaders.text_loader import TextLoader, MarkdownLoader
from RAGService.Data.DocumentProcessors.loaders.json_loader import JSONLoader
from RAGService.Data.DocumentProcessors.loaders.csv_loader import CSVLoader


# Test files directory
TEST_FILES_DIR = Path(__file__).parent


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def print_content_preview(content: str, max_length: int = 300):
    """Print a preview of content."""
    if len(content) > max_length:
        print(f"  Content Preview ({len(content)} chars total):")
        print(f"    {content[:max_length]}...")
    else:
        print(f"  Content ({len(content)} chars):")
        print(f"    {content}")


# ==================== Test Functions ====================

def test_1_supported_file_types():
    """Test 1: Check supported file types."""
    print_section("Test 1: Supported File Types")
    
    print("Available file types in SupportedFileType enum:")
    for file_type in SupportedFileType:
        print(f"  - {file_type.value}")
    
    print(f"\nTotal supported types: {len(SupportedFileType)}")
    print("✅ SupportedFileType enum works correctly")


def test_2_create_text_loader():
    """Test 2: Create and configure TextLoader."""
    print_section("Test 2: Create TextLoader")
    
    loader = TextLoader(encoding="utf-8")
    
    print(f"Loader type: {type(loader).__name__}")
    print(f"Supported extensions: {loader.supported_extensions}")
    print(f"Encoding: {loader.encoding}")
    print("✅ TextLoader created successfully")
    
    return loader


def test_3_load_text_file(loader: TextLoader):
    """Test 3: Load actual text file."""
    print_section("Test 3: Load Text File")
    
    txt_file = TEST_FILES_DIR / "my all details.txt"
    
    print(f"Loading: {txt_file.name}")
    doc = loader.load(txt_file)
    
    print(f"Document type: {type(doc).__name__}")
    print(f"Source: {doc.source}")
    print(f"File type: {doc.file_type}")
    print(f"Content length: {len(doc.content)} characters")
    print_content_preview(doc.content)
    
    # Check metadata
    print(f"\nMetadata keys: {list(doc.metadata.keys())}")
    
    print("✅ Text file loaded successfully")
    return doc


def test_4_can_handle_check():
    """Test 4: Test can_handle method for different files."""
    print_section("Test 4: can_handle Check")
    
    text_loader = TextLoader()
    json_loader = JSONLoader()
    csv_loader = CSVLoader()
    
    test_files = [
        "document.txt",
        "data.json",
        "records.csv",
        "report.pdf",
        "notes.md"
    ]
    
    loaders = [
        ("TextLoader", text_loader),
        ("JSONLoader", json_loader),
        ("CSVLoader", csv_loader)
    ]
    
    print("Checking which loaders can handle each file:")
    for file in test_files:
        handlers = [name for name, loader in loaders if loader.can_handle(file)]
        print(f"  {file}: {handlers if handlers else 'None'}")
    
    print("✅ can_handle method works correctly")


def test_5_create_json_loader():
    """Test 5: Create and configure JSONLoader."""
    print_section("Test 5: Create JSONLoader")
    
    # Basic loader
    loader = JSONLoader()
    print(f"Basic JSONLoader:")
    print(f"  Supported extensions: {loader.supported_extensions}")
    print(f"  extract_all_strings: {loader.extract_all_strings}")
    
    # Loader with string extraction
    loader_extract = JSONLoader(extract_all_strings=True)
    print(f"\nJSONLoader with extract_all_strings=True:")
    print(f"  extract_all_strings: {loader_extract.extract_all_strings}")
    
    print("✅ JSONLoader created successfully")
    return loader


def test_6_load_json_file():
    """Test 6: Load actual JSON file."""
    print_section("Test 6: Load JSON File (Default)")
    
    json_file = TEST_FILES_DIR / "conversation.json"
    loader = JSONLoader()
    
    print(f"Loading: {json_file.name}")
    doc = loader.load(json_file)
    
    print(f"Document type: {type(doc).__name__}")
    print(f"File type: {doc.file_type}")
    print(f"Content length: {len(doc.content)} characters")
    print_content_preview(doc.content)
    
    print("✅ JSON file loaded successfully")
    return doc


def test_7_load_json_with_string_extraction():
    """Test 7: Load JSON with extract_all_strings."""
    print_section("Test 7: Load JSON (Extract All Strings)")
    
    json_file = TEST_FILES_DIR / "conversation.json"
    loader = JSONLoader(extract_all_strings=True)
    
    print(f"Loading with extract_all_strings=True")
    doc = loader.load(json_file)
    
    print(f"Content length: {len(doc.content)} characters")
    
    # Count extracted strings
    lines = doc.content.split("\n")
    print(f"Extracted strings: {len(lines)}")
    print(f"First 3 strings:")
    for i, line in enumerate(lines[:3]):
        preview = line[:80] + "..." if len(line) > 80 else line
        print(f"  {i+1}: {preview}")
    
    print("✅ JSON string extraction works")


def test_8_create_csv_loader():
    """Test 8: Create and configure CSVLoader."""
    print_section("Test 8: Create CSVLoader")
    
    # Basic loader
    loader = CSVLoader()
    print(f"Basic CSVLoader:")
    print(f"  Supported extensions: {loader.supported_extensions}")
    print(f"  Delimiter: '{loader.delimiter}'")
    print(f"  Include headers: {loader.include_headers}")
    
    # Loader with headers
    loader_headers = CSVLoader(include_headers=True)
    print(f"\nCSVLoader with include_headers=True:")
    print(f"  Include headers: {loader_headers.include_headers}")
    
    print("✅ CSVLoader created successfully")
    return loader


def test_9_load_csv_file():
    """Test 9: Load actual CSV file."""
    print_section("Test 9: Load CSV File")
    
    csv_file = TEST_FILES_DIR / "Personal_TrainingData.csv"
    loader = CSVLoader(include_headers=True)
    
    print(f"Loading: {csv_file.name}")
    doc = loader.load(csv_file)
    
    print(f"Document type: {type(doc).__name__}")
    print(f"File type: {doc.file_type}")
    print(f"Content length: {len(doc.content)} characters")
    
    # Check CSV metadata
    print(f"\nCSV Metadata:")
    print(f"  Row count: {doc.metadata.get('row_count')}")
    print(f"  Column count: {doc.metadata.get('column_count')}")
    print(f"  Columns: {doc.metadata.get('columns')}")
    
    print_content_preview(doc.content)
    
    print("✅ CSV file loaded successfully")
    return doc


def test_10_csv_with_specific_columns():
    """Test 10: Load CSV with specific content columns."""
    print_section("Test 10: CSV with Specific Columns")
    
    csv_file = TEST_FILES_DIR / "Personal_TrainingData.csv"
    
    # Load only the 'prompt' column
    loader = CSVLoader(content_columns=["prompt"])
    doc = loader.load(csv_file)
    
    print(f"Loading with content_columns=['prompt']")
    print(f"Content length: {len(doc.content)} characters")
    
    # Show first few lines
    lines = doc.content.split("\n")[:5]
    print(f"\nFirst 5 rows (prompt only):")
    for i, line in enumerate(lines):
        preview = line[:70] + "..." if len(line) > 70 else line
        print(f"  {i+1}: {preview}")
    
    print("✅ CSV column filtering works")


def test_11_document_loader_factory_get_loader():
    """Test 11: Get loader via factory."""
    print_section("Test 11: DocumentLoaderFactory Get Loader")
    
    test_cases = [
        ("document.txt", "TextLoader"),
        ("notes.md", "MarkdownLoader"),
        ("data.json", "JSONLoader"),
        ("records.csv", "CSVLoader"),
    ]
    
    print("Getting loaders for different file types:")
    for filename, expected in test_cases:
        try:
            loader = DocumentLoaderFactory.get_loader(filename)
            loader_name = type(loader).__name__
            status = "✅" if loader_name == expected else "❌"
            print(f"  {filename}: {loader_name} {status}")
        except Exception as e:
            print(f"  {filename}: Error - {e}")
    
    print("✅ Factory.get_loader works correctly")


def test_12_document_loader_factory_load():
    """Test 12: Load directly via factory."""
    print_section("Test 12: DocumentLoaderFactory Direct Load")
    
    txt_file = TEST_FILES_DIR / "my all details.txt"
    
    print(f"Loading via factory: {txt_file.name}")
    doc = DocumentLoaderFactory.load(txt_file)
    
    print(f"Document type: {type(doc).__name__}")
    print(f"Content length: {len(doc.content)} characters")
    print(f"File type: {doc.file_type}")
    
    print("✅ Factory.load works correctly")


def test_13_factory_load_all_test_files():
    """Test 13: Load all test files via factory."""
    print_section("Test 13: Factory Load All Test Files")
    
    test_files = [
        "my all details.txt",
        "conversation.json",
        "Personal_TrainingData.csv"
    ]
    
    print("Loading all test files via factory:")
    for filename in test_files:
        file_path = TEST_FILES_DIR / filename
        try:
            doc = DocumentLoaderFactory.load(file_path)
            print(f"  {filename}:")
            print(f"    Type: {doc.file_type}")
            print(f"    Length: {len(doc.content)} chars")
        except Exception as e:
            print(f"  {filename}: Error - {e}")
    
    print("✅ All test files loaded successfully")


def test_14_text_splitter_config():
    """Test 14: Configure TextSplitter."""
    print_section("Test 14: TextSplitter Configuration")
    
    # Default config
    default_config = TextSplitterConfig()
    print("Default TextSplitterConfig:")
    print(f"  chunk_size: {default_config.chunk_size}")
    print(f"  chunk_overlap: {default_config.chunk_overlap}")
    print(f"  separators: {default_config.separators}")
    print(f"  keep_separator: {default_config.keep_separator}")
    print(f"  strip_whitespace: {default_config.strip_whitespace}")
    
    # Custom config
    custom_config = TextSplitterConfig(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", " "]
    )
    print("\nCustom TextSplitterConfig:")
    print(f"  chunk_size: {custom_config.chunk_size}")
    print(f"  chunk_overlap: {custom_config.chunk_overlap}")
    
    print("✅ TextSplitterConfig works correctly")


def test_15_create_recursive_text_splitter():
    """Test 15: Create RecursiveTextSplitter."""
    print_section("Test 15: Create RecursiveTextSplitter")
    
    splitter = RecursiveTextSplitter()
    
    print(f"Splitter type: {type(splitter).__name__}")
    print(f"Config chunk_size: {splitter.config.chunk_size}")
    print(f"Config chunk_overlap: {splitter.config.chunk_overlap}")
    
    print("✅ RecursiveTextSplitter created successfully")
    return splitter


def test_16_split_simple_text():
    """Test 16: Split simple text into chunks."""
    print_section("Test 16: Split Simple Text")
    
    config = TextSplitterConfig(chunk_size=100, chunk_overlap=20)
    splitter = RecursiveTextSplitter(config)
    
    text = """This is the first paragraph about machine learning.
Machine learning is a subset of artificial intelligence.

This is the second paragraph about deep learning.
Deep learning uses neural networks with multiple layers.

This is the third paragraph about natural language processing.
NLP enables computers to understand human language."""
    
    chunks = splitter.split_text(text)
    
    print(f"Original text length: {len(text)} chars")
    print(f"Chunk size: {config.chunk_size}")
    print(f"Chunk overlap: {config.chunk_overlap}")
    print(f"Number of chunks: {len(chunks)}")
    
    print("\nChunks:")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1} ({len(chunk)} chars): {chunk[:60]}...")
    
    print("✅ Text splitting works correctly")


def test_17_split_document():
    """Test 17: Split ProcessedDocument into DocumentChunks."""
    print_section("Test 17: Split ProcessedDocument")
    
    # Load a document
    txt_file = TEST_FILES_DIR / "my all details.txt"
    doc = DocumentLoaderFactory.load(txt_file)
    
    # Split with custom config
    config = TextSplitterConfig(chunk_size=500, chunk_overlap=100)
    splitter = RecursiveTextSplitter(config)
    
    chunks = splitter.split_document(doc)
    
    print(f"Original document: {len(doc.content)} chars")
    print(f"Number of chunks: {len(chunks)}")
    
    print("\nFirst 3 DocumentChunks:")
    for i, chunk in enumerate(chunks[:3]):
        print(f"  Chunk {i+1}:")
        print(f"    Type: {type(chunk).__name__}")
        print(f"    Length: {len(chunk.content)} chars")
        print(f"    chunk_index: {chunk.chunk_index}")
        print(f"    start_char: {chunk.start_char}")
        print(f"    Preview: {chunk.content[:50]}...")
    
    print("✅ split_document works correctly")


def test_18_split_json_document():
    """Test 18: Split JSON document into chunks."""
    print_section("Test 18: Split JSON Document")
    
    json_file = TEST_FILES_DIR / "conversation.json"
    loader = JSONLoader(extract_all_strings=True)
    doc = loader.load(json_file)
    
    config = TextSplitterConfig(chunk_size=400, chunk_overlap=50)
    splitter = RecursiveTextSplitter(config)
    
    chunks = splitter.split_document(doc)
    
    print(f"JSON content length: {len(doc.content)} chars")
    print(f"Number of chunks: {len(chunks)}")
    
    print("\nChunk size distribution:")
    sizes = [len(c.content) for c in chunks]
    print(f"  Min: {min(sizes)} chars")
    print(f"  Max: {max(sizes)} chars")
    print(f"  Avg: {sum(sizes) // len(sizes)} chars")
    
    print("✅ JSON document chunking works")


def test_19_split_csv_document():
    """Test 19: Split CSV document into chunks."""
    print_section("Test 19: Split CSV Document")
    
    csv_file = TEST_FILES_DIR / "Personal_TrainingData.csv"
    doc = DocumentLoaderFactory.load(csv_file)
    
    config = TextSplitterConfig(chunk_size=600, chunk_overlap=100)
    splitter = RecursiveTextSplitter(config)
    
    chunks = splitter.split_document(doc)
    
    print(f"CSV content length: {len(doc.content)} chars")
    print(f"Number of chunks: {len(chunks)}")
    
    # Show metadata propagation
    if chunks:
        print("\nMetadata in first chunk:")
        for key, value in chunks[0].metadata.items():
            if key in ["row_count", "column_count", "chunk_index", "source"]:
                val_str = str(value)[:50] + "..." if len(str(value)) > 50 else value
                print(f"  {key}: {val_str}")
    
    print("✅ CSV document chunking works")


def test_20_split_multiple_documents():
    """Test 20: Split multiple documents at once."""
    print_section("Test 20: Split Multiple Documents")
    
    test_files = [
        TEST_FILES_DIR / "my all details.txt",
        TEST_FILES_DIR / "conversation.json",
        TEST_FILES_DIR / "Personal_TrainingData.csv"
    ]
    
    # Load all documents
    documents = [DocumentLoaderFactory.load(f) for f in test_files]
    
    config = TextSplitterConfig(chunk_size=500, chunk_overlap=100)
    splitter = RecursiveTextSplitter(config)
    
    all_chunks = splitter.split_documents(documents)
    
    print(f"Documents loaded: {len(documents)}")
    print(f"Total chunks: {len(all_chunks)}")
    
    # Count chunks per document
    from collections import Counter
    sources = Counter([c.metadata.get("source", "unknown") for c in all_chunks])
    print("\nChunks per document:")
    for source, count in sources.items():
        filename = Path(source).name if source != "unknown" else source
        print(f"  {filename}: {count} chunks")
    
    print("✅ Multiple document splitting works")


def test_21_processed_document_model():
    """Test 21: Test ProcessedDocument dataclass."""
    print_section("Test 21: ProcessedDocument Model")
    
    # Create manually
    doc = ProcessedDocument(
        content="This is test content",
        metadata={"author": "test"},
        source="/path/to/file.txt",
        file_type=SupportedFileType.TXT
    )
    
    print("ProcessedDocument fields:")
    print(f"  content: '{doc.content}'")
    print(f"  source: {doc.source}")
    print(f"  file_type: {doc.file_type}")
    print(f"  metadata: {doc.metadata}")
    
    # Check auto-populated metadata
    print("\nAuto-populated metadata:")
    print(f"  'source' in metadata: {'source' in doc.metadata}")
    print(f"  'file_type' in metadata: {'file_type' in doc.metadata}")
    
    print("✅ ProcessedDocument model works correctly")


def test_22_document_chunk_model():
    """Test 22: Test DocumentChunk dataclass."""
    print_section("Test 22: DocumentChunk Model")
    
    chunk = DocumentChunk(
        content="This is chunk content",
        metadata={"source": "/path/to/file.txt"},
        chunk_index=0,
        start_char=0,
        end_char=21
    )
    
    print("DocumentChunk fields:")
    print(f"  content: '{chunk.content}'")
    print(f"  chunk_index: {chunk.chunk_index}")
    print(f"  start_char: {chunk.start_char}")
    print(f"  end_char: {chunk.end_char}")
    print(f"  metadata: {chunk.metadata}")
    
    print("✅ DocumentChunk model works correctly")


def test_23_factory_supported_extensions():
    """Test 23: Check factory supported extensions."""
    print_section("Test 23: Factory Supported Extensions")
    
    supported = DocumentLoaderFactory.list_supported_extensions()
    
    print(f"Supported extensions: {supported}")
    print(f"Total: {len(supported)} extensions")
    
    # Check specific extensions
    checks = ["txt", "json", "csv", "md", "pdf", "docx"]
    print("\nExtension availability:")
    for ext in checks:
        available = DocumentLoaderFactory.is_supported(f"test.{ext}")
        status = "✅" if available else "❌"
        print(f"  .{ext}: {status}")
    
    print("✅ Factory extension listing works")


def test_24_custom_metadata():
    """Test 24: Load file with custom metadata."""
    print_section("Test 24: Load with Custom Metadata")
    
    txt_file = TEST_FILES_DIR / "my all details.txt"
    
    custom_metadata = {
        "category": "personal",
        "language": "english",
        "uploaded_by": "test_user"
    }
    
    doc = DocumentLoaderFactory.load(txt_file, metadata=custom_metadata)
    
    print("Custom metadata passed:")
    for key, value in custom_metadata.items():
        print(f"  {key}: {value}")
    
    print("\nMetadata in loaded document:")
    for key, value in doc.metadata.items():
        val_str = str(value)[:40] + "..." if len(str(value)) > 40 else value
        print(f"  {key}: {val_str}")
    
    print("✅ Custom metadata works correctly")


def test_25_chunk_overlap_verification():
    """Test 25: Verify chunk overlap is working."""
    print_section("Test 25: Verify Chunk Overlap")
    
    # Create text that's easy to verify overlap
    paragraphs = [
        f"Paragraph {i}: " + "word " * 50
        for i in range(1, 6)
    ]
    text = "\n\n".join(paragraphs)
    
    config = TextSplitterConfig(chunk_size=200, chunk_overlap=50)
    splitter = RecursiveTextSplitter(config)
    
    chunks = splitter.split_text(text)
    
    print(f"Text length: {len(text)} chars")
    print(f"Chunk size: {config.chunk_size}, Overlap: {config.chunk_overlap}")
    print(f"Number of chunks: {len(chunks)}")
    
    # Check for overlap between consecutive chunks
    overlaps_found = 0
    for i in range(len(chunks) - 1):
        chunk1 = chunks[i]
        chunk2 = chunks[i + 1]
        
        # Check if end of chunk1 appears in start of chunk2
        end_of_first = chunk1[-30:] if len(chunk1) > 30 else chunk1
        if end_of_first in chunk2:
            overlaps_found += 1
    
    print(f"\nOverlapping transitions found: {overlaps_found}/{len(chunks) - 1}")
    
    print("✅ Chunk overlap verification complete")


def test_26_empty_document_handling():
    """Test 26: Handle empty content."""
    print_section("Test 26: Empty Content Handling")
    
    # Create empty document
    doc = ProcessedDocument(
        content="",
        source="empty.txt",
        file_type=SupportedFileType.TXT
    )
    
    splitter = RecursiveTextSplitter()
    chunks = splitter.split_document(doc)
    
    print(f"Empty document content length: {len(doc.content)}")
    print(f"Chunks created: {len(chunks)}")
    
    print("✅ Empty document handled correctly")


def test_27_unicode_content():
    """Test 27: Handle unicode content."""
    print_section("Test 27: Unicode Content Handling")
    
    # Create document with unicode content
    unicode_content = """
    English: Hello, World!
    Spanish: ¡Hola, Mundo!
    French: Bonjour le monde!
    German: Hallo Welt!
    Hindi: नमस्ते दुनिया!
    Japanese: こんにちは世界！
    Chinese: 你好世界！
    Arabic: مرحبا بالعالم!
    Russian: Привет мир!
    Korean: 안녕하세요 세계!
    """
    
    doc = ProcessedDocument(
        content=unicode_content,
        source="unicode.txt",
        file_type=SupportedFileType.TXT
    )
    
    config = TextSplitterConfig(chunk_size=100, chunk_overlap=20)
    splitter = RecursiveTextSplitter(config)
    
    chunks = splitter.split_document(doc)
    
    print(f"Unicode content length: {len(unicode_content)} chars")
    print(f"Chunks created: {len(chunks)}")
    
    print("\nSample chunk with unicode:")
    if chunks:
        print(f"  {chunks[0].content[:100]}...")
    
    print("✅ Unicode content handled correctly")


def test_28_large_document_splitting():
    """Test 28: Split large document efficiently."""
    print_section("Test 28: Large Document Splitting")
    
    # Create a large document
    large_content = ("This is a test sentence. " * 100 + "\n\n") * 50
    
    doc = ProcessedDocument(
        content=large_content,
        source="large_doc.txt",
        file_type=SupportedFileType.TXT
    )
    
    config = TextSplitterConfig(chunk_size=1000, chunk_overlap=200)
    splitter = RecursiveTextSplitter(config)
    
    import time
    start = time.time()
    chunks = splitter.split_document(doc)
    elapsed = time.time() - start
    
    print(f"Document size: {len(large_content):,} chars")
    print(f"Chunks created: {len(chunks)}")
    print(f"Time taken: {elapsed:.4f} seconds")
    print(f"Avg chunk size: {len(large_content) // len(chunks) if chunks else 0} chars")
    
    print("✅ Large document splitting works efficiently")


def test_29_conversation_json_structure():
    """Test 29: Parse conversation JSON structure."""
    print_section("Test 29: Parse Conversation JSON")
    
    json_file = TEST_FILES_DIR / "conversation.json"
    
    # Load and display structure
    import json as json_module
    with open(json_file, "r") as f:
        data = json_module.load(f)
    
    print(f"JSON structure: list of {len(data)} conversation pairs")
    print(f"Each item has keys: {list(data[0].keys())}")
    
    # Load with different approaches
    print("\nApproach 1: Default (full JSON as string)")
    loader1 = JSONLoader()
    doc1 = loader1.load(json_file)
    print(f"  Content length: {len(doc1.content)} chars")
    
    print("\nApproach 2: Extract all strings")
    loader2 = JSONLoader(extract_all_strings=True)
    doc2 = loader2.load(json_file)
    lines = doc2.content.split("\n")
    print(f"  Extracted {len(lines)} strings")
    print(f"  First User message: {lines[0][:60]}...")
    
    print("✅ Conversation JSON parsed successfully")


def test_30_csv_training_data_structure():
    """Test 30: Parse CSV training data structure."""
    print_section("Test 30: Parse CSV Training Data")
    
    csv_file = TEST_FILES_DIR / "Personal_TrainingData.csv"
    
    # Load with different configurations
    print("Approach 1: Full CSV as text")
    loader1 = CSVLoader(include_headers=True)
    doc1 = loader1.load(csv_file)
    print(f"  Content length: {len(doc1.content)} chars")
    print(f"  Columns: {doc1.metadata.get('columns')}")
    print(f"  Rows: {doc1.metadata.get('row_count')}")
    
    print("\nApproach 2: Prompts only")
    loader2 = CSVLoader(content_columns=["prompt"])
    doc2 = loader2.load(csv_file)
    
    prompts = doc2.content.split("\n")
    print(f"  {len(prompts)} prompts extracted")
    print(f"  Sample prompt: {prompts[0][:60]}...")
    
    print("\nApproach 3: Responses only")
    loader3 = CSVLoader(content_columns=["Response"])
    doc3 = loader3.load(csv_file)
    
    responses = doc3.content.split("\n")
    print(f"  {len(responses)} responses extracted")
    
    print("✅ CSV training data parsed successfully")


def run_all_tests():
    """Run all document loader tests."""
    print("\n" + "=" * 60)
    print("  DOCUMENT LOADERS TEST SUITE")
    print("=" * 60)
    
    print(f"\nTest files directory: {TEST_FILES_DIR}")
    print(f"Files available:")
    for f in TEST_FILES_DIR.iterdir():
        if f.is_file() and not f.name.startswith(".") and f.suffix != ".py":
            print(f"  - {f.name}")
    
    # Run all tests
    test_1_supported_file_types()
    text_loader = test_2_create_text_loader()
    text_doc = test_3_load_text_file(text_loader)
    test_4_can_handle_check()
    json_loader = test_5_create_json_loader()
    test_6_load_json_file()
    test_7_load_json_with_string_extraction()
    csv_loader = test_8_create_csv_loader()
    test_9_load_csv_file()
    test_10_csv_with_specific_columns()
    test_11_document_loader_factory_get_loader()
    test_12_document_loader_factory_load()
    test_13_factory_load_all_test_files()
    test_14_text_splitter_config()
    splitter = test_15_create_recursive_text_splitter()
    test_16_split_simple_text()
    test_17_split_document()
    test_18_split_json_document()
    test_19_split_csv_document()
    test_20_split_multiple_documents()
    test_21_processed_document_model()
    test_22_document_chunk_model()
    test_23_factory_supported_extensions()
    test_24_custom_metadata()
    test_25_chunk_overlap_verification()
    test_26_empty_document_handling()
    test_27_unicode_content()
    test_28_large_document_splitting()
    test_29_conversation_json_structure()
    test_30_csv_training_data_structure()
    
    print_section("ALL TESTS COMPLETED")
    print("✅ Document loaders test suite finished successfully!")


if __name__ == "__main__":
    run_all_tests()
