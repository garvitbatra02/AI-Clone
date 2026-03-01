"""
Row Chunker

Chunking strategy for CSV files.
Each row becomes one chunk, formatted as key-value pairs.
Optionally groups N rows per chunk for very short rows.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from RAGService.Data.DocumentProcessors.base import ProcessedDocument
from RAGService.Data.DocumentProcessors.strategies.base_strategy import (
    BaseChunkingStrategy,
)
from RAGService.Data.VectorDB.base import DocumentChunk


class RowChunker(BaseChunkingStrategy):
    """
    Chunker for CSV files — each row becomes one chunk.
    
    Row content is formatted as key-value pairs:
        "name: John\\nage: 30\\ncity: New York"
    
    Metadata includes row_start, row_end, and columns list.
    
    For very short rows, rows_per_chunk can group multiple rows
    into a single chunk.
    """
    
    def __init__(
        self,
        max_chunk_size: int = 1000,
        chunk_overlap: int = 0,
        rows_per_chunk: int = 1,
    ):
        """
        Args:
            max_chunk_size: Max chars per chunk (only relevant when grouping rows).
            chunk_overlap: Not used for row-based chunking.
            rows_per_chunk: Number of rows to group into one chunk.
        """
        super().__init__(max_chunk_size, chunk_overlap)
        self.rows_per_chunk = max(1, rows_per_chunk)
    
    def chunk(
        self,
        document: ProcessedDocument,
        structural_map: Optional[Any] = None,
    ) -> List[DocumentChunk]:
        """
        Chunk a CSV document — one (or N) row(s) per chunk.
        
        Uses rows_data and column_names from loader metadata.
        Falls back to line-based splitting if raw data not available.
        """
        metadata = document.metadata
        rows_data = metadata.get("rows_data")
        column_names = metadata.get("column_names") or metadata.get("columns", [])
        
        if rows_data:
            chunks = self._chunk_from_rows(document, rows_data, column_names)
        else:
            # Fallback: split content by lines
            chunks = self._chunk_from_lines(document)
        
        # Post-pass
        for i, chunk in enumerate(chunks):
            chunk.chunk_index = i
            chunk.metadata["chunk_index"] = i
        self._stamp_total_chunks(chunks)
        
        return chunks
    
    def _chunk_from_rows(
        self,
        document: ProcessedDocument,
        rows_data: List[Dict[str, str]],
        column_names: List[str],
    ) -> List[DocumentChunk]:
        """Create chunks from raw row data."""
        chunks = []
        
        for i in range(0, len(rows_data), self.rows_per_chunk):
            batch = rows_data[i:i + self.rows_per_chunk]
            row_start = i + 1  # 1-based (excluding header)
            row_end = i + len(batch)
            
            # Format rows as key-value pairs
            row_texts = []
            for row in batch:
                parts = []
                for col in column_names:
                    value = row.get(col, "")
                    if value:
                        parts.append(f"{col}: {value}")
                row_texts.append("\n".join(parts))
            
            content = "\n---\n".join(row_texts) if len(row_texts) > 1 else row_texts[0]
            
            meta = self._build_base_metadata(
                document,
                content,
                chunk_index=len(chunks),
                row_start=row_start,
                row_end=row_end,
                columns=column_names,
            )
            
            chunks.append(self._make_chunk(
                content=content,
                metadata=meta,
                chunk_index=len(chunks),
                source=document.source,
            ))
        
        return chunks
    
    def _chunk_from_lines(self, document: ProcessedDocument) -> List[DocumentChunk]:
        """Fallback: split CSV content by lines."""
        lines = document.content.strip().split("\n")
        columns = document.metadata.get("columns", [])
        
        # Skip header line if it matches column names
        start_idx = 0
        if lines and columns:
            header = lines[0]
            if any(col in header for col in columns[:2]):
                start_idx = 1
        
        chunks = []
        for i in range(start_idx, len(lines), self.rows_per_chunk):
            batch = lines[i:i + self.rows_per_chunk]
            content = "\n".join(batch)
            
            if not content.strip():
                continue
            
            row_start = i - start_idx + 1
            row_end = row_start + len(batch) - 1
            
            meta = self._build_base_metadata(
                document,
                content,
                chunk_index=len(chunks),
                row_start=row_start,
                row_end=row_end,
                columns=columns,
            )
            
            chunks.append(self._make_chunk(
                content=content,
                metadata=meta,
                chunk_index=len(chunks),
                source=document.source,
            ))
        
        return chunks
