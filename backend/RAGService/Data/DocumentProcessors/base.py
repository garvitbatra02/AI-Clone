"""
Document Processors Base Module

This module provides the core abstractions for document loading and processing:
- BaseDocumentLoader: Abstract base class for document loaders
- TextSplitter: Utility for splitting documents into chunks
- ProcessedDocument: Model for loaded documents
- DocumentChunk: Imported from VectorDB for unified chunk representation
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Union

from RAGService.Data.VectorDB.base import DocumentChunk


class SupportedFileType(str, Enum):
    """Supported file types for document loading."""
    TXT = "txt"
    MD = "md"
    MARKDOWN = "markdown"
    PDF = "pdf"
    JSON = "json"
    CSV = "csv"
    DOCX = "docx"
    HTML = "html"
    XML = "xml"


@dataclass
class ProcessedDocument:
    """
    A processed document with content and metadata.
    
    Attributes:
        content: The full text content of the document
        metadata: Arbitrary metadata associated with the document
        source: Source file path or URL
        file_type: Type of the source file
    """
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None
    file_type: Optional[SupportedFileType] = None
    
    def __post_init__(self):
        if self.source and "source" not in self.metadata:
            self.metadata["source"] = self.source
        if self.file_type and "file_type" not in self.metadata:
            self.metadata["file_type"] = self.file_type.value if self.file_type else None


class BaseDocumentLoader(ABC):
    """
    Abstract base class for document loaders.
    
    Document loaders are responsible for reading and parsing
    documents from various file formats into ProcessedDocument objects.
    """
    
    @property
    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """List of file extensions this loader supports (without dot)."""
        pass
    
    @abstractmethod
    def load(
        self,
        source: Union[str, Path, BinaryIO],
        metadata: Optional[Dict[str, Any]] = None
    ) -> ProcessedDocument:
        """
        Load and parse a document.
        
        Args:
            source: File path, URL, or file-like object
            metadata: Optional metadata to include
            
        Returns:
            ProcessedDocument with content and metadata
        """
        pass
    
    @abstractmethod
    async def async_load(
        self,
        source: Union[str, Path, BinaryIO],
        metadata: Optional[Dict[str, Any]] = None
    ) -> ProcessedDocument:
        """Async version of load."""
        pass
    
    def can_handle(self, file_path: Union[str, Path]) -> bool:
        """Check if this loader can handle the given file."""
        path = Path(file_path)
        extension = path.suffix.lower().lstrip(".")
        return extension in self.supported_extensions
    
    def _get_file_type(self, file_path: Union[str, Path]) -> Optional[SupportedFileType]:
        """Get the SupportedFileType for a file."""
        path = Path(file_path)
        extension = path.suffix.lower().lstrip(".")
        try:
            return SupportedFileType(extension)
        except ValueError:
            return None


class TextSplitterConfig:
    """Configuration for text splitting."""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
        keep_separator: bool = True,
        strip_whitespace: bool = True,
        add_start_index: bool = True
    ):
        """
        Initialize text splitter configuration.
        
        Args:
            chunk_size: Maximum size of chunks (in characters)
            chunk_overlap: Overlap between chunks (in characters)
            separators: List of separators to split on (in order of priority)
            keep_separator: Whether to keep separators in chunks
            strip_whitespace: Whether to strip whitespace from chunks
            add_start_index: Whether to track start index in original document
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]
        self.keep_separator = keep_separator
        self.strip_whitespace = strip_whitespace
        self.add_start_index = add_start_index


class RecursiveTextSplitter:
    """
    Recursively splits text into chunks using multiple separators.
    
    This is the recommended splitter for most use cases. It tries to
    split on paragraph breaks first, then sentences, then words.
    
    Based on LangChain's RecursiveCharacterTextSplitter.
    """
    
    def __init__(self, config: Optional[TextSplitterConfig] = None):
        """
        Initialize the text splitter.
        
        Args:
            config: Splitter configuration (uses defaults if not provided)
        """
        self.config = config or TextSplitterConfig()
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks.
        
        Args:
            text: The text to split
            
        Returns:
            List of text chunks
        """
        return self._split_text(text, self.config.separators)
    
    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text using separators."""
        final_chunks = []
        
        # Get the appropriate separator
        separator = separators[-1]  # Default to last (empty string)
        new_separators = []
        
        for i, sep in enumerate(separators):
            if sep == "":
                separator = sep
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break
        
        # Split the text
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)
        
        # Process splits
        good_splits = []
        separator_to_use = separator if self.config.keep_separator else ""
        
        for split in splits:
            if len(split) < self.config.chunk_size:
                good_splits.append(split)
            else:
                # Recursively split large chunks
                if good_splits:
                    merged = self._merge_splits(good_splits, separator_to_use)
                    final_chunks.extend(merged)
                    good_splits = []
                
                if new_separators:
                    other_chunks = self._split_text(split, new_separators)
                    final_chunks.extend(other_chunks)
                else:
                    final_chunks.append(split)
        
        if good_splits:
            merged = self._merge_splits(good_splits, separator_to_use)
            final_chunks.extend(merged)
        
        return final_chunks
    
    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """Merge splits into chunks of appropriate size with overlap."""
        chunks = []
        current_chunk: List[str] = []
        current_length = 0
        
        for split in splits:
            split_length = len(split)
            
            # Check if adding this split would exceed chunk size
            if current_length + split_length + len(separator) > self.config.chunk_size:
                if current_chunk:
                    chunk_text = separator.join(current_chunk)
                    if self.config.strip_whitespace:
                        chunk_text = chunk_text.strip()
                    if chunk_text:
                        chunks.append(chunk_text)
                    
                    # Handle overlap
                    while (
                        current_chunk and
                        current_length > self.config.chunk_overlap
                    ):
                        removed = current_chunk.pop(0)
                        current_length -= len(removed) + len(separator)
            
            current_chunk.append(split)
            current_length += split_length + len(separator)
        
        # Add final chunk
        if current_chunk:
            chunk_text = separator.join(current_chunk)
            if self.config.strip_whitespace:
                chunk_text = chunk_text.strip()
            if chunk_text:
                chunks.append(chunk_text)
        
        return chunks
    
    def split_document(
        self,
        document: ProcessedDocument
    ) -> List[DocumentChunk]:
        """
        Split a ProcessedDocument into DocumentChunks.
        
        Args:
            document: The document to split
            
        Returns:
            List of DocumentChunk objects
        """
        text = document.content
        chunks = self.split_text(text)
        
        doc_chunks = []
        current_index = 0
        
        for i, chunk_text in enumerate(chunks):
            # Find the start position in original text
            if self.config.add_start_index:
                start_char = text.find(chunk_text, current_index)
                if start_char == -1:
                    start_char = current_index
                end_char = start_char + len(chunk_text)
                current_index = start_char + 1
            else:
                start_char = None
                end_char = None
            
            # Copy and extend metadata
            chunk_metadata = document.metadata.copy()
            chunk_metadata["chunk_index"] = i
            if start_char is not None:
                chunk_metadata["start_char"] = start_char
                chunk_metadata["end_char"] = end_char
            
            doc_chunks.append(DocumentChunk(
                content=chunk_text,
                metadata=chunk_metadata,
                chunk_index=i,
                start_char=start_char,
                end_char=end_char,
                source=document.source
            ))
        
        return doc_chunks
    
    def split_documents(
        self,
        documents: List[ProcessedDocument]
    ) -> List[DocumentChunk]:
        """
        Split multiple documents into chunks.
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of all DocumentChunks from all documents
        """
        all_chunks = []
        for doc in documents:
            chunks = self.split_document(doc)
            all_chunks.extend(chunks)
        return all_chunks
