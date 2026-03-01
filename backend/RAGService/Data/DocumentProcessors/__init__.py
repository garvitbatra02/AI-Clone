"""
DocumentProcessors Module

Provides document loading, parsing, text splitting, and smart chunking.
"""

from RAGService.Data.DocumentProcessors.base import (
    BaseDocumentLoader,
    DocumentChunk,
    ProcessedDocument,
    RecursiveTextSplitter,
    SupportedFileType,
    TextSplitterConfig,
)
from RAGService.Data.DocumentProcessors.loader_factory import (
    DocumentLoaderFactory,
)
from RAGService.Data.DocumentProcessors.chunk_metadata import (
    ChunkMetadata,
)
from RAGService.Data.DocumentProcessors.smart_chunker import (
    SmartChunker,
    SmartChunkerConfig,
)
from RAGService.Data.DocumentProcessors.strategies import (
    BaseChunkingStrategy,
    SemanticChunker,
    TextChunker,
    RowChunker,
    JsonEntryChunker,
    LLMStructureAnalyzer,
    StructuralElement,
    StructuralMap,
)

__all__ = [
    # Base classes and models
    "BaseDocumentLoader",
    "ProcessedDocument",
    "DocumentChunk",
    "SupportedFileType",
    # Text splitting (legacy)
    "RecursiveTextSplitter",
    "TextSplitterConfig",
    # Factory
    "DocumentLoaderFactory",
    # Metadata
    "ChunkMetadata",
    # Smart chunking
    "SmartChunker",
    "SmartChunkerConfig",
    # Strategies
    "BaseChunkingStrategy",
    "SemanticChunker",
    "TextChunker",
    "RowChunker",
    "JsonEntryChunker",
    "LLMStructureAnalyzer",
    "StructuralElement",
    "StructuralMap",
]
