"""
Base Chunking Strategy

Abstract base class for all format-specific chunking strategies.
Provides shared metadata building utilities.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from RAGService.Data.DocumentProcessors.base import ProcessedDocument
from RAGService.Data.DocumentProcessors.chunk_metadata import ChunkMetadata
from RAGService.Data.VectorDB.base import DocumentChunk


class BaseChunkingStrategy(ABC):
    """
    Abstract base for format-specific chunking strategies.
    
    Subclasses implement chunk() to split a ProcessedDocument into
    DocumentChunks with rich ChunkMetadata. The base class provides
    shared helpers for metadata building and token estimation.
    """
    
    def __init__(self, max_chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Args:
            max_chunk_size: Maximum characters per chunk.
            chunk_overlap: Character overlap when falling back to recursive splitting.
        """
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
    
    @abstractmethod
    def chunk(
        self,
        document: ProcessedDocument,
        structural_map: Optional[Any] = None,
    ) -> List[DocumentChunk]:
        """
        Split a document into chunks with rich metadata.
        
        Args:
            document: The loaded document to chunk.
            structural_map: Optional LLM-generated structural analysis.
            
        Returns:
            List of DocumentChunk objects with ChunkMetadata in metadata.
        """
        pass
    
    def _build_base_metadata(
        self,
        document: ProcessedDocument,
        chunk_text: str,
        chunk_index: int,
        total_chunks: int = 0,
        **format_specific,
    ) -> Dict[str, Any]:
        """
        Build a ChunkMetadata dict for a chunk.
        
        Populates universal fields and merges in format-specific fields.
        Returns a flat dict ready for DocumentChunk.metadata.
        
        Args:
            document: Source document.
            chunk_text: The chunk content (for token estimation).
            chunk_index: Zero-based chunk index.
            total_chunks: Total chunks from this document (filled in post-pass).
            **format_specific: Format-specific fields (page_number, topic, etc.)
        """
        source = document.source or ""
        file_type = document.file_type.value if document.file_type else ""
        
        meta = ChunkMetadata(
            source_file=source,
            file_type=file_type,
            source_modified_at=ChunkMetadata.get_file_modified_time(source) if source else None,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            token_count=ChunkMetadata.estimate_tokens(chunk_text),
        )
        
        # Apply format-specific fields
        for key, value in format_specific.items():
            if hasattr(meta, key) and value is not None:
                setattr(meta, key, value)
        
        return meta.to_dict()
    
    def _make_chunk(
        self,
        content: str,
        metadata: Dict[str, Any],
        chunk_index: int,
        source: Optional[str] = None,
        start_char: Optional[int] = None,
        end_char: Optional[int] = None,
    ) -> DocumentChunk:
        """Create a DocumentChunk with the given attributes."""
        return DocumentChunk(
            content=content,
            metadata=metadata,
            chunk_index=chunk_index,
            source=source,
            start_char=start_char,
            end_char=end_char,
        )
    
    def _stamp_total_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Post-pass: set total_chunks on all chunk metadata."""
        total = len(chunks)
        for chunk in chunks:
            chunk.metadata["total_chunks"] = total
