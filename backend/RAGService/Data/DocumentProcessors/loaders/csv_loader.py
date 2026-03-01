"""
CSV Document Loader

Handles CSV files, converting rows to text for embedding.
"""

from __future__ import annotations

import asyncio
import csv
import io
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Union

import aiofiles

from RAGService.Data.DocumentProcessors.base import (
    BaseDocumentLoader,
    ProcessedDocument,
    SupportedFileType,
)


class CSVLoader(BaseDocumentLoader):
    """
    Loader for CSV files.
    
    Converts CSV rows to text in various formats:
    - Row-per-line: Each row becomes a line of text
    - Key-value: Each row becomes "column: value" pairs
    - Custom template: User-defined format string
    """
    
    def __init__(
        self,
        content_columns: Optional[List[str]] = None,
        metadata_columns: Optional[List[str]] = None,
        row_template: Optional[str] = None,
        delimiter: str = ",",
        encoding: str = "utf-8",
        include_headers: bool = False
    ):
        """
        Initialize the CSV loader.
        
        Args:
            content_columns: Columns to include in content (None = all)
            metadata_columns: Columns to extract as document metadata
            row_template: Template for formatting rows (e.g., "{name}: {value}")
            delimiter: CSV delimiter character
            encoding: Text encoding
            include_headers: Whether to include column headers in content
        """
        self.content_columns = content_columns
        self.metadata_columns = metadata_columns or []
        self.row_template = row_template
        self.delimiter = delimiter
        self.encoding = encoding
        self.include_headers = include_headers
    
    @property
    def supported_extensions(self) -> List[str]:
        return ["csv"]
    
    def _format_row(self, row: Dict[str, str], columns: List[str]) -> str:
        """Format a row as text."""
        if self.row_template:
            try:
                return self.row_template.format(**row)
            except KeyError:
                pass
        
        # Default: key-value format
        parts = []
        for col in columns:
            if col in row:
                parts.append(f"{col}: {row[col]}")
        return " | ".join(parts)
    
    def _process_csv(self, reader: csv.DictReader) -> tuple[str, Dict[str, Any], List[str]]:
        """Process CSV data into content and metadata."""
        rows = list(reader)
        
        if not rows:
            return "", {}, []
        
        # Determine which columns to use
        all_columns = list(reader.fieldnames or [])
        columns = self.content_columns or all_columns
        
        # Build content
        lines = []
        
        if self.include_headers:
            lines.append(" | ".join(columns))
            lines.append("-" * 40)
        
        for row in rows:
            line = self._format_row(row, columns)
            lines.append(line)
        
        content = "\n".join(lines)
        
        # Extract metadata — include raw rows for downstream RowChunker
        metadata = {
            "row_count": len(rows),
            "column_count": len(all_columns),
            "columns": all_columns,
            "column_names": columns,  # columns used for content
            "rows_data": rows,  # raw row dicts for per-row chunking
        }
        
        if self.metadata_columns and rows:
            for col in self.metadata_columns:
                if col in rows[0]:
                    metadata[col] = rows[0][col]
        
        return content, metadata, all_columns
    
    def load(
        self,
        source: Union[str, Path, BinaryIO],
        metadata: Optional[Dict[str, Any]] = None
    ) -> ProcessedDocument:
        """Load a CSV file."""
        meta = metadata or {}
        
        if isinstance(source, (str, Path)):
            path = Path(source)
            with open(path, "r", encoding=self.encoding, newline="") as f:
                reader = csv.DictReader(f, delimiter=self.delimiter)
                content, csv_meta, _ = self._process_csv(reader)
            source_str = str(path.absolute())
        else:
            content_bytes = source.read()
            if isinstance(content_bytes, bytes):
                content_bytes = content_bytes.decode(self.encoding)
            reader = csv.DictReader(
                io.StringIO(content_bytes),
                delimiter=self.delimiter
            )
            content, csv_meta, _ = self._process_csv(reader)
            source_str = meta.get("source", "unknown")
        
        meta.update(csv_meta)
        
        return ProcessedDocument(
            content=content,
            metadata=meta,
            source=source_str,
            file_type=SupportedFileType.CSV
        )
    
    async def async_load(
        self,
        source: Union[str, Path, BinaryIO],
        metadata: Optional[Dict[str, Any]] = None
    ) -> ProcessedDocument:
        """Async version of load."""
        meta = metadata or {}
        
        if isinstance(source, (str, Path)):
            path = Path(source)
            async with aiofiles.open(path, "r", encoding=self.encoding, newline="") as f:
                content_str = await f.read()
            reader = csv.DictReader(
                io.StringIO(content_str),
                delimiter=self.delimiter
            )
            content, csv_meta, _ = self._process_csv(reader)
            source_str = str(path.absolute())
        else:
            content_bytes = source.read()
            if isinstance(content_bytes, bytes):
                content_bytes = content_bytes.decode(self.encoding)
            reader = csv.DictReader(
                io.StringIO(content_bytes),
                delimiter=self.delimiter
            )
            content, csv_meta, _ = self._process_csv(reader)
            source_str = meta.get("source", "unknown")
        
        meta.update(csv_meta)
        
        return ProcessedDocument(
            content=content,
            metadata=meta,
            source=source_str,
            file_type=SupportedFileType.CSV
        )
