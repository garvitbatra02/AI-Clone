"""
Text Document Loader

Handles plain text files (.txt) and markdown files (.md, .markdown).
"""

from __future__ import annotations

import aiofiles
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Union

from RAGService.Data.DocumentProcessors.base import (
    BaseDocumentLoader,
    ProcessedDocument,
    SupportedFileType,
)


class TextLoader(BaseDocumentLoader):
    """
    Loader for plain text files.
    
    Supports .txt files with various encodings.
    """
    
    def __init__(self, encoding: str = "utf-8"):
        """
        Initialize the text loader.
        
        Args:
            encoding: Text encoding to use (default: utf-8)
        """
        self.encoding = encoding
    
    @property
    def supported_extensions(self) -> List[str]:
        return ["txt"]
    
    def load(
        self,
        source: Union[str, Path, BinaryIO],
        metadata: Optional[Dict[str, Any]] = None
    ) -> ProcessedDocument:
        """Load a text file."""
        meta = metadata or {}
        
        if isinstance(source, (str, Path)):
            path = Path(source)
            with open(path, "r", encoding=self.encoding) as f:
                content = f.read()
            source_str = str(path.absolute())
            file_type = self._get_file_type(path)
        else:
            # File-like object
            content = source.read()
            if isinstance(content, bytes):
                content = content.decode(self.encoding)
            source_str = meta.get("source", "unknown")
            file_type = SupportedFileType.TXT
        
        return ProcessedDocument(
            content=content,
            metadata=meta,
            source=source_str,
            file_type=file_type
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
            async with aiofiles.open(path, "r", encoding=self.encoding) as f:
                content = await f.read()
            source_str = str(path.absolute())
            file_type = self._get_file_type(path)
        else:
            # File-like objects are handled synchronously
            content = source.read()
            if isinstance(content, bytes):
                content = content.decode(self.encoding)
            source_str = meta.get("source", "unknown")
            file_type = SupportedFileType.TXT
        
        return ProcessedDocument(
            content=content,
            metadata=meta,
            source=source_str,
            file_type=file_type
        )


class MarkdownLoader(BaseDocumentLoader):
    """
    Loader for Markdown files.
    
    Supports .md and .markdown files. Optionally strips markdown
    formatting to extract plain text.
    """
    
    def __init__(
        self,
        encoding: str = "utf-8",
        strip_formatting: bool = False
    ):
        """
        Initialize the markdown loader.
        
        Args:
            encoding: Text encoding to use
            strip_formatting: Whether to strip markdown formatting
        """
        self.encoding = encoding
        self.strip_formatting = strip_formatting
    
    @property
    def supported_extensions(self) -> List[str]:
        return ["md", "markdown"]
    
    def _strip_markdown(self, content: str) -> str:
        """Strip markdown formatting from content."""
        import re
        
        # Remove headers
        content = re.sub(r'^#{1,6}\s+', '', content, flags=re.MULTILINE)
        
        # Remove bold/italic
        content = re.sub(r'\*\*(.+?)\*\*', r'\1', content)
        content = re.sub(r'\*(.+?)\*', r'\1', content)
        content = re.sub(r'__(.+?)__', r'\1', content)
        content = re.sub(r'_(.+?)_', r'\1', content)
        
        # Remove links, keep text
        content = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', content)
        
        # Remove images
        content = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'\1', content)
        
        # Remove code blocks
        content = re.sub(r'```[\s\S]*?```', '', content)
        content = re.sub(r'`(.+?)`', r'\1', content)
        
        # Remove horizontal rules
        content = re.sub(r'^[-*_]{3,}\s*$', '', content, flags=re.MULTILINE)
        
        # Remove blockquotes
        content = re.sub(r'^>\s+', '', content, flags=re.MULTILINE)
        
        return content.strip()
    
    def load(
        self,
        source: Union[str, Path, BinaryIO],
        metadata: Optional[Dict[str, Any]] = None
    ) -> ProcessedDocument:
        """Load a markdown file."""
        meta = metadata or {}
        
        if isinstance(source, (str, Path)):
            path = Path(source)
            with open(path, "r", encoding=self.encoding) as f:
                content = f.read()
            source_str = str(path.absolute())
            file_type = self._get_file_type(path)
        else:
            content = source.read()
            if isinstance(content, bytes):
                content = content.decode(self.encoding)
            source_str = meta.get("source", "unknown")
            file_type = SupportedFileType.MD
        
        if self.strip_formatting:
            content = self._strip_markdown(content)
        
        return ProcessedDocument(
            content=content,
            metadata=meta,
            source=source_str,
            file_type=file_type
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
            async with aiofiles.open(path, "r", encoding=self.encoding) as f:
                content = await f.read()
            source_str = str(path.absolute())
            file_type = self._get_file_type(path)
        else:
            content = source.read()
            if isinstance(content, bytes):
                content = content.decode(self.encoding)
            source_str = meta.get("source", "unknown")
            file_type = SupportedFileType.MD
        
        if self.strip_formatting:
            content = self._strip_markdown(content)
        
        return ProcessedDocument(
            content=content,
            metadata=meta,
            source=source_str,
            file_type=file_type
        )
