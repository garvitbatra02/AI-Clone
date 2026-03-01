"""
JSON Entry Chunker

Chunking strategy for JSON files.
Each top-level entry (array element or object key) becomes one chunk.
Metadata includes the json_path for each entry.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from RAGService.Data.DocumentProcessors.base import ProcessedDocument
from RAGService.Data.DocumentProcessors.strategies.base_strategy import (
    BaseChunkingStrategy,
)
from RAGService.Data.VectorDB.base import DocumentChunk


class JsonEntryChunker(BaseChunkingStrategy):
    """
    Chunker for JSON files — each entry becomes one chunk.
    
    For arrays: each element is a chunk with json_path like "[0]", "[1]".
    For objects: each top-level key is a chunk with json_path like "users", "settings".
    
    Entry content is serialized as formatted JSON or key-value text,
    depending on the entry structure.
    """
    
    def __init__(
        self,
        max_chunk_size: int = 2000,
        chunk_overlap: int = 0,
        flatten_simple: bool = True,
    ):
        """
        Args:
            max_chunk_size: Max chars per chunk (large entries may exceed this).
            chunk_overlap: Not used for entry-based chunking.
            flatten_simple: If True, simple key-value dicts are formatted as
                           "key: value" pairs instead of JSON.
        """
        super().__init__(max_chunk_size, chunk_overlap)
        self.flatten_simple = flatten_simple
    
    def chunk(
        self,
        document: ProcessedDocument,
        structural_map: Optional[Any] = None,
    ) -> List[DocumentChunk]:
        """
        Chunk a JSON document — one entry per chunk.
        
        Uses entries_data and entry_paths from loader metadata.
        Falls back to treating entire content as one chunk if
        structural data is not available.
        """
        metadata = document.metadata
        entries_data = metadata.get("entries_data")
        entry_paths = metadata.get("entry_paths")
        
        if entries_data and entry_paths:
            chunks = self._chunk_from_entries(
                document, entries_data, entry_paths
            )
        else:
            # Fallback: entire JSON as one chunk
            chunks = self._chunk_whole(document)
        
        # Post-pass
        for i, chunk in enumerate(chunks):
            chunk.chunk_index = i
            chunk.metadata["chunk_index"] = i
        self._stamp_total_chunks(chunks)
        
        return chunks
    
    def _chunk_from_entries(
        self,
        document: ProcessedDocument,
        entries_data: List[Any],
        entry_paths: List[str],
    ) -> List[DocumentChunk]:
        """Create one chunk per JSON entry."""
        chunks = []
        
        for i, (entry, path) in enumerate(zip(entries_data, entry_paths)):
            content = self._format_entry(entry)
            
            if not content.strip():
                continue
            
            meta = self._build_base_metadata(
                document,
                content,
                chunk_index=i,
                json_path=path,
            )
            
            chunks.append(self._make_chunk(
                content=content,
                metadata=meta,
                chunk_index=i,
                source=document.source,
            ))
        
        return chunks
    
    def _chunk_whole(self, document: ProcessedDocument) -> List[DocumentChunk]:
        """Fallback: treat entire JSON content as one chunk."""
        content = document.content
        meta = self._build_base_metadata(
            document,
            content,
            chunk_index=0,
        )
        return [self._make_chunk(
            content=content,
            metadata=meta,
            chunk_index=0,
            source=document.source,
        )]
    
    def _format_entry(self, entry: Any) -> str:
        """
        Format a JSON entry as readable text.
        
        Simple flat dicts → key-value pairs.
        Complex/nested structures → formatted JSON.
        """
        if self.flatten_simple and isinstance(entry, dict):
            # Check if all values are simple (str, int, float, bool, None)
            if all(
                isinstance(v, (str, int, float, bool, type(None)))
                for v in entry.values()
            ):
                parts = []
                for k, v in entry.items():
                    parts.append(f"{k}: {v}")
                return "\n".join(parts)
        
        if isinstance(entry, str):
            return entry
        
        return json.dumps(entry, indent=2, ensure_ascii=False)
