"""
Chunk Metadata Schema

Defines the standardized metadata schema for document chunks across
all file formats. Ensures consistent, rich metadata for every chunk
stored in the vector database.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class ChunkMetadata:
    """
    Standardized metadata for a document chunk.
    
    Universal fields are populated for all file types.
    Format-specific fields are nullable and only populated
    when relevant to the source format.
    
    Attributes:
        # --- Universal (always populated) ---
        source_file: Absolute path or identifier of the source document
        file_type: File format (pdf, docx, txt, csv, json)
        ingested_at: ISO timestamp when the chunk was created
        source_modified_at: ISO timestamp of source file's last modification
        chunk_index: Zero-based index of this chunk within the document
        total_chunks: Total number of chunks from the same document
        token_count: Approximate token count of the chunk content
        language: Language code (e.g. "en", "hi", "es") or None if unknown
        
        # --- PDF / DOCX specific ---
        page_number: Page number (1-based) for PDF chunks
        section_title: Immediate heading/section title
        heading_path: Full heading hierarchy (e.g. ["Chapter 1", "1.2 Methods"])
        has_table: Whether this chunk contains tabular data
        has_code: Whether this chunk contains code blocks
        
        # --- CSV specific ---
        row_start: Starting row number (1-based, excluding header)
        row_end: Ending row number (inclusive)
        columns: List of column names in this chunk
        
        # --- JSON specific ---
        json_path: Dot-notation path to the entry (e.g. "data[0]", "users[2].profile")
        
        # --- TXT specific (LLM-assisted) ---
        topic: Topic label assigned by LLM topical grouping
        
        # --- Application-level (user-provided) ---
        extra: Open dict for user-defined fields (category, tags, access_level, version, etc.)
    """
    
    # Universal fields
    source_file: str = ""
    file_type: str = ""
    ingested_at: str = ""
    source_modified_at: Optional[str] = None
    chunk_index: int = 0
    total_chunks: int = 0
    token_count: int = 0
    language: Optional[str] = None
    
    # PDF / DOCX specific
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    heading_path: Optional[List[str]] = None
    has_table: bool = False
    has_code: bool = False
    
    # CSV specific
    row_start: Optional[int] = None
    row_end: Optional[int] = None
    columns: Optional[List[str]] = None
    
    # JSON specific
    json_path: Optional[str] = None
    
    # TXT specific (LLM-assisted)
    topic: Optional[str] = None
    
    # Application-level (open)
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.ingested_at:
            self.ingested_at = datetime.now(timezone.utc).isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to a flat dictionary, stripping None values.
        
        Produces a clean payload suitable for Qdrant storage.
        Lists and nested values are preserved (Qdrant supports them).
        The 'extra' dict is merged into the top level.
        """
        result: Dict[str, Any] = {}
        
        # Universal fields (always present)
        result["source_file"] = self.source_file
        result["file_type"] = self.file_type
        result["ingested_at"] = self.ingested_at
        result["chunk_index"] = self.chunk_index
        result["total_chunks"] = self.total_chunks
        result["token_count"] = self.token_count
        
        # Nullable fields — only include if set
        if self.source_modified_at is not None:
            result["source_modified_at"] = self.source_modified_at
        if self.language is not None:
            result["language"] = self.language
        if self.page_number is not None:
            result["page_number"] = self.page_number
        if self.section_title is not None:
            result["section_title"] = self.section_title
        if self.heading_path is not None:
            result["heading_path"] = self.heading_path
        if self.has_table:
            result["has_table"] = True
        if self.has_code:
            result["has_code"] = True
        if self.row_start is not None:
            result["row_start"] = self.row_start
        if self.row_end is not None:
            result["row_end"] = self.row_end
        if self.columns is not None:
            result["columns"] = self.columns
        if self.json_path is not None:
            result["json_path"] = self.json_path
        if self.topic is not None:
            result["topic"] = self.topic
        
        # Merge extra fields into top level
        if self.extra:
            result.update(self.extra)
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChunkMetadata":
        """
        Reconstruct ChunkMetadata from a flat dictionary.
        
        Known fields are extracted; remaining fields go into 'extra'.
        """
        known_fields = {
            "source_file", "file_type", "ingested_at", "source_modified_at",
            "chunk_index", "total_chunks", "token_count", "language",
            "page_number", "section_title", "heading_path", "has_table",
            "has_code", "row_start", "row_end", "columns", "json_path",
            "topic",
        }
        
        kwargs = {}
        extra = {}
        
        for key, value in data.items():
            if key in known_fields:
                kwargs[key] = value
            else:
                extra[key] = value
        
        kwargs["extra"] = extra
        return cls(**kwargs)
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        Estimate token count for a text string.
        
        Uses tiktoken if available, otherwise falls back to
        a word-based approximation (~0.75 tokens per word).
        """
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except (ImportError, Exception):
            # Fallback: rough approximation
            return int(len(text.split()) * 1.3)
    
    @staticmethod
    def get_file_modified_time(file_path: str) -> Optional[str]:
        """Get the last modification time of a file as ISO string."""
        import os
        try:
            mtime = os.path.getmtime(file_path)
            return datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
        except (OSError, ValueError):
            return None
