"""
Smart Chunker

Central dispatcher that routes documents to the correct format-specific
chunking strategy. Optionally invokes an LLM for structural analysis
before chunking.

Usage:
    from RAGService.Data.DocumentProcessors.smart_chunker import (
        SmartChunker,
        SmartChunkerConfig,
    )
    
    # Without LLM
    chunker = SmartChunker()
    chunks = chunker.chunk_file("report.pdf")
    
    # With LLM analysis
    config = SmartChunkerConfig(
        use_llm_analysis=True,
        llm_provider="groq",
        llm_model="llama-3.1-8b-instant",
        llm_api_keys=["key1"],
    )
    chunker = SmartChunker(config)
    chunks = chunker.chunk_file("report.pdf")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from RAGService.Data.DocumentProcessors.base import (
    ProcessedDocument,
    SupportedFileType,
)
from RAGService.Data.DocumentProcessors.loader_factory import (
    DocumentLoaderFactory,
)
from RAGService.Data.DocumentProcessors.strategies.base_strategy import (
    BaseChunkingStrategy,
)
from RAGService.Data.DocumentProcessors.strategies.semantic_chunker import (
    SemanticChunker,
)
from RAGService.Data.DocumentProcessors.strategies.text_chunker import (
    TextChunker,
)
from RAGService.Data.DocumentProcessors.strategies.row_chunker import (
    RowChunker,
)
from RAGService.Data.DocumentProcessors.strategies.json_entry_chunker import (
    JsonEntryChunker,
)
from RAGService.Data.VectorDB.base import DocumentChunk

logger = logging.getLogger(__name__)


# ==================== Configuration ====================

@dataclass
class SmartChunkerConfig:
    """
    Configuration for SmartChunker.
    
    Attributes:
        max_chunk_size: Global default max chunk size (characters).
        chunk_overlap: Global default overlap between chunks.
        use_llm_analysis: Whether to invoke LLM structural analysis.
        llm_provider: LLM provider name ("groq", "cerebras", "gemini").
        llm_model: LLM model name (e.g. "llama-3.1-8b-instant").
        llm_api_keys: API keys for the LLM provider.
        llm_temperature: Temperature for LLM calls (0.0 recommended).
        
        Per-format overrides (None = use global defaults):
        pdf_max_chunk_size: Override for PDF files.
        docx_max_chunk_size: Override for DOCX files.
        txt_max_chunk_size: Override for TXT files.
        csv_rows_per_chunk: Rows per chunk for CSV (default 1).
    """
    # Global defaults
    max_chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # LLM settings
    use_llm_analysis: bool = False
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    llm_api_keys: Optional[List[str]] = None
    llm_temperature: float = 0.0
    
    # Per-format overrides
    pdf_max_chunk_size: Optional[int] = None
    docx_max_chunk_size: Optional[int] = None
    txt_max_chunk_size: Optional[int] = None
    csv_rows_per_chunk: int = 1


# ==================== File Type Mapping ====================

# Map SupportedFileType values to strategy categories
_FORMAT_CATEGORY = {
    SupportedFileType.PDF: "semantic",
    SupportedFileType.DOCX: "semantic",
    SupportedFileType.TXT: "text",
    SupportedFileType.MD: "text",
    SupportedFileType.MARKDOWN: "text",
    SupportedFileType.CSV: "csv",
    SupportedFileType.JSON: "json",
}

# LLM analysis modes per category
_LLM_MODE = {
    "semantic": "structural",
    "text": "topical",
}


# ==================== SmartChunker ====================

class SmartChunker:
    """
    Central dispatcher for format-aware document chunking.
    
    Routes each file to the correct strategy based on its type:
    - PDF / DOCX → SemanticChunker (heading/section-aware)
    - TXT / MD   → TextChunker (paragraph / topic-aware)
    - CSV         → RowChunker (row-per-chunk)
    - JSON        → JsonEntryChunker (entry-per-chunk)
    
    Optionally invokes LLMStructureAnalyzer before chunking to
    provide boundary hints to SemanticChunker and TextChunker.
    
    Example:
        chunker = SmartChunker()
        chunks = chunker.chunk_file("report.pdf")
        
        # With config
        config = SmartChunkerConfig(max_chunk_size=800)
        chunker = SmartChunker(config)
        chunks = chunker.chunk_file("data.csv")
    """
    
    def __init__(self, config: Optional[SmartChunkerConfig] = None):
        self.config = config or SmartChunkerConfig()
        self._strategies: Dict[str, BaseChunkingStrategy] = {}
        self._llm_analyzer = None
        
        # Build strategy instances
        self._init_strategies()
        
        # Build LLM analyzer if configured
        if self.config.use_llm_analysis:
            self._init_llm_analyzer()
    
    # ==================== Public API ====================
    
    def chunk_file(
        self,
        file_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
        **loader_kwargs,
    ) -> List[DocumentChunk]:
        """
        Load a file and chunk it using the appropriate strategy.
        
        Args:
            file_path: Path to the file.
            metadata: Extra metadata to merge into loader metadata.
            **loader_kwargs: Passed through to the document loader.
            
        Returns:
            List of DocumentChunk with embeddings=None (caller embeds).
        """
        path = Path(file_path)
        
        # 1. Load document
        doc = DocumentLoaderFactory.load(path, metadata, **loader_kwargs)
        
        # 2. Determine file type
        file_type = doc.file_type or self._detect_file_type(path)
        
        # 3. Chunk
        return self.chunk_document(doc, file_type)
    
    async def async_chunk_file(
        self,
        file_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
        **loader_kwargs,
    ) -> List[DocumentChunk]:
        """Async version of chunk_file."""
        path = Path(file_path)
        
        doc = await DocumentLoaderFactory.async_load(path, metadata, **loader_kwargs)
        file_type = doc.file_type or self._detect_file_type(path)
        
        return await self.async_chunk_document(doc, file_type)
    
    def chunk_document(
        self,
        document: ProcessedDocument,
        file_type: Optional[SupportedFileType] = None,
    ) -> List[DocumentChunk]:
        """
        Chunk a pre-loaded ProcessedDocument.
        
        Args:
            document: The loaded document.
            file_type: File type (auto-detected from metadata if None).
            
        Returns:
            List of DocumentChunk.
        """
        file_type = file_type or self._type_from_metadata(document)
        category = _FORMAT_CATEGORY.get(file_type, "text")
        strategy = self._strategies.get(category)
        
        if strategy is None:
            logger.warning(f"No strategy for category '{category}', falling back to text")
            strategy = self._strategies["text"]
        
        # Optional LLM analysis
        structural_map = None
        if self._llm_analyzer and category in _LLM_MODE:
            mode = _LLM_MODE[category]
            structural_map = self._llm_analyzer.analyze(
                document.content, mode=mode
            )
            if structural_map:
                logger.info(
                    f"LLM structural map: {len(structural_map.elements)} elements, "
                    f"coverage={structural_map.coverage:.2f}"
                )
        
        chunks = strategy.chunk(document, structural_map=structural_map)
        return chunks
    
    async def async_chunk_document(
        self,
        document: ProcessedDocument,
        file_type: Optional[SupportedFileType] = None,
    ) -> List[DocumentChunk]:
        """Async version of chunk_document."""
        file_type = file_type or self._type_from_metadata(document)
        category = _FORMAT_CATEGORY.get(file_type, "text")
        strategy = self._strategies.get(category)
        
        if strategy is None:
            strategy = self._strategies["text"]
        
        structural_map = None
        if self._llm_analyzer and category in _LLM_MODE:
            mode = _LLM_MODE[category]
            structural_map = await self._llm_analyzer.async_analyze(
                document.content, mode=mode
            )
            if structural_map:
                logger.info(
                    f"LLM structural map: {len(structural_map.elements)} elements, "
                    f"coverage={structural_map.coverage:.2f}"
                )
        
        # Strategies themselves are synchronous (no I/O in chunking logic)
        chunks = strategy.chunk(document, structural_map=structural_map)
        return chunks
    
    # ==================== Initialization ====================
    
    def _init_strategies(self) -> None:
        """Create strategy instances for each category."""
        cfg = self.config
        
        self._strategies["semantic"] = SemanticChunker(
            max_chunk_size=cfg.pdf_max_chunk_size or cfg.max_chunk_size,
            chunk_overlap=cfg.chunk_overlap,
        )
        
        self._strategies["text"] = TextChunker(
            max_chunk_size=cfg.txt_max_chunk_size or cfg.max_chunk_size,
            chunk_overlap=cfg.chunk_overlap,
        )
        
        self._strategies["csv"] = RowChunker(
            max_chunk_size=cfg.max_chunk_size,
            chunk_overlap=0,
            rows_per_chunk=cfg.csv_rows_per_chunk,
        )
        
        self._strategies["json"] = JsonEntryChunker(
            max_chunk_size=cfg.max_chunk_size,
            chunk_overlap=0,
        )
    
    def _init_llm_analyzer(self) -> None:
        """Create the LLM analyzer from config."""
        try:
            from ChatService.Chat.llm import LLMFactory, LLMProvider
            from RAGService.Data.DocumentProcessors.strategies.llm_analyzer import (
                LLMStructureAnalyzer,
            )
            
            provider_str = (self.config.llm_provider or "groq").lower()
            provider = LLMProvider(provider_str)
            
            llm = LLMFactory.create(
                provider=provider,
                model=self.config.llm_model or "llama-3.1-8b-instant",
                api_keys=self.config.llm_api_keys or [],
                temperature=self.config.llm_temperature,
            )
            
            self._llm_analyzer = LLMStructureAnalyzer(llm=llm)
            logger.info(
                f"LLM structure analyzer initialized: "
                f"provider={provider_str}, model={self.config.llm_model}"
            )
        except Exception as e:
            logger.warning(f"Failed to initialize LLM analyzer: {e}")
            self._llm_analyzer = None
    
    # ==================== Helpers ====================
    
    @staticmethod
    def _detect_file_type(path: Path) -> Optional[SupportedFileType]:
        """Detect file type from extension."""
        ext = path.suffix.lower().lstrip(".")
        try:
            return SupportedFileType(ext)
        except ValueError:
            return None
    
    @staticmethod
    def _type_from_metadata(document: ProcessedDocument) -> Optional[SupportedFileType]:
        """Extract file type from document metadata or file_type field."""
        if document.file_type:
            return document.file_type
        
        ft = document.metadata.get("file_type")
        if ft:
            try:
                return SupportedFileType(ft)
            except ValueError:
                pass
        
        source = document.source or document.metadata.get("source", "")
        if source:
            ext = Path(source).suffix.lower().lstrip(".")
            try:
                return SupportedFileType(ext)
            except ValueError:
                pass
        
        return None
