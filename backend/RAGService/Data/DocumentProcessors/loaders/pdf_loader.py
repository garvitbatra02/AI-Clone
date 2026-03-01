"""
PDF Document Loader

Handles PDF files using pypdf or pdfplumber.Emits page_boundaries metadata for downstream chunkers to assign page numbers."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Union

from RAGService.Data.DocumentProcessors.base import (
    BaseDocumentLoader,
    ProcessedDocument,
    SupportedFileType,
)


class PDFLoader(BaseDocumentLoader):
    """
    Loader for PDF files.
    
    Uses pypdf by default for text extraction. Can optionally
    use pdfplumber for better table handling.
    
    Emits structural metadata:
        - page_count: Total number of pages
        - page_boundaries: List of (start_char, end_char) tuples per page
        - pdf_info: PDF document properties (pypdf only)
    
    Note: Requires pypdf package: pip install pypdf
    For better table extraction: pip install pdfplumber
    """
    
    def __init__(
        self,
        extract_images: bool = False,
        use_pdfplumber: bool = False,
        password: Optional[str] = None
    ):
        """
        Initialize the PDF loader.
        
        Args:
            extract_images: Whether to attempt OCR on images (requires additional deps)
            use_pdfplumber: Use pdfplumber instead of pypdf for extraction
            password: Password for encrypted PDFs
        """
        self.extract_images = extract_images
        self.use_pdfplumber = use_pdfplumber
        self.password = password
    
    @property
    def supported_extensions(self) -> List[str]:
        return ["pdf"]
    
    def _extract_with_pypdf(
        self,
        file_path: Union[str, Path, BinaryIO]
    ) -> tuple[str, Dict[str, Any]]:
        """Extract text using pypdf with page boundary tracking."""
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError(
                "pypdf is required for PDF loading. "
                "Install with: pip install pypdf"
            )
        
        if isinstance(file_path, (str, Path)):
            reader = PdfReader(str(file_path), password=self.password)
        else:
            reader = PdfReader(file_path, password=self.password)
        
        pages = []
        page_boundaries = []  # (start_char, end_char) per page
        current_offset = 0
        
        for page in reader.pages:
            text = page.extract_text()
            if text:
                start = current_offset
                pages.append(text)
                # Account for the "\n\n" join separator between pages
                current_offset += len(text) + 2  # +2 for "\n\n"
                end = current_offset - 2  # end is exclusive of separator
                page_boundaries.append((start, end))
        
        content = "\n\n".join(pages)
        metadata = {
            "page_count": len(reader.pages),
            "page_boundaries": page_boundaries,
            "pdf_info": reader.metadata.__dict__ if reader.metadata else {}
        }
        
        return content, metadata
    
    def _extract_with_pdfplumber(
        self,
        file_path: Union[str, Path, BinaryIO]
    ) -> tuple[str, Dict[str, Any]]:
        """Extract text using pdfplumber with page boundary tracking."""
        try:
            import pdfplumber
        except ImportError:
            raise ImportError(
                "pdfplumber is required for enhanced PDF loading. "
                "Install with: pip install pdfplumber"
            )
        
        pages = []
        page_boundaries = []
        current_offset = 0
        page_count = 0
        
        with pdfplumber.open(file_path, password=self.password) as pdf:
            page_count = len(pdf.pages)
            for page in pdf.pages:
                page_parts = []
                
                text = page.extract_text()
                if text:
                    page_parts.append(text)
                
                # Extract tables separately — marked with [TABLE] tags
                tables = page.extract_tables()
                for table in tables:
                    if table:
                        table_text = self._format_table(table)
                        page_parts.append(f"[TABLE]\n{table_text}\n[/TABLE]")
                
                if page_parts:
                    page_text = "\n\n".join(page_parts)
                    start = current_offset
                    pages.append(page_text)
                    current_offset += len(page_text) + 2
                    end = current_offset - 2
                    page_boundaries.append((start, end))
        
        content = "\n\n".join(pages)
        metadata = {
            "page_count": page_count,
            "page_boundaries": page_boundaries,
        }
        
        return content, metadata
    
    def _format_table(self, table: List[List]) -> str:
        """Format a table as text."""
        rows = []
        for row in table:
            if row:
                cells = [str(cell) if cell else "" for cell in row]
                rows.append(" | ".join(cells))
        return "\n".join(rows)
    
    def load(
        self,
        source: Union[str, Path, BinaryIO],
        metadata: Optional[Dict[str, Any]] = None
    ) -> ProcessedDocument:
        """Load a PDF file."""
        meta = metadata or {}
        
        if self.use_pdfplumber:
            content, pdf_meta = self._extract_with_pdfplumber(source)
        else:
            content, pdf_meta = self._extract_with_pypdf(source)
        
        # Merge metadata
        meta.update(pdf_meta)
        
        if isinstance(source, (str, Path)):
            source_str = str(Path(source).absolute())
        else:
            source_str = meta.get("source", "unknown")
        
        return ProcessedDocument(
            content=content,
            metadata=meta,
            source=source_str,
            file_type=SupportedFileType.PDF
        )
    
    async def async_load(
        self,
        source: Union[str, Path, BinaryIO],
        metadata: Optional[Dict[str, Any]] = None
    ) -> ProcessedDocument:
        """
        Async version of load.
        
        Note: PDF parsing is CPU-bound, so we run it in a thread pool.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.load(source, metadata)
        )
