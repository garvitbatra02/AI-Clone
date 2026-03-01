"""
JSON Document Loader

Handles JSON files, extracting text content from various structures.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Union

import aiofiles

from RAGService.Data.DocumentProcessors.base import (
    BaseDocumentLoader,
    ProcessedDocument,
    SupportedFileType,
)


class JSONLoader(BaseDocumentLoader):
    """
    Loader for JSON files.
    
    Can extract text from JSON in various ways:
    - Full JSON as formatted string
    - Specific keys (jq-style path)
    - Recursive text extraction from all string values
    """
    
    def __init__(
        self,
        content_key: Optional[str] = None,
        metadata_keys: Optional[List[str]] = None,
        extract_all_strings: bool = False,
        encoding: str = "utf-8"
    ):
        """
        Initialize the JSON loader.
        
        Args:
            content_key: JSON key/path to extract content from (e.g., "data.text")
            metadata_keys: Keys to extract as metadata
            extract_all_strings: Recursively extract all string values
            encoding: Text encoding
        """
        self.content_key = content_key
        self.metadata_keys = metadata_keys or []
        self.extract_all_strings = extract_all_strings
        self.encoding = encoding
    
    @property
    def supported_extensions(self) -> List[str]:
        return ["json"]
    
    def _get_nested_value(self, data: Dict, key_path: str) -> Any:
        """Get a value from nested dict using dot notation."""
        keys = key_path.split(".")
        value = data
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            elif isinstance(value, list) and key.isdigit():
                value = value[int(key)]
            else:
                return None
        return value
    
    def _extract_all_strings(self, data: Any, strings: Optional[List[str]] = None) -> List[str]:
        """Recursively extract all string values from a structure."""
        if strings is None:
            strings = []
        
        if isinstance(data, str):
            strings.append(data)
        elif isinstance(data, dict):
            for value in data.values():
                self._extract_all_strings(value, strings)
        elif isinstance(data, list):
            for item in data:
                self._extract_all_strings(item, strings)
        
        return strings
    
    def _process_json(self, data: Any) -> tuple[str, Dict[str, Any]]:
        """Process JSON data into content and metadata."""
        metadata = {}
        
        # Extract metadata
        if isinstance(data, dict):
            for key in self.metadata_keys:
                value = self._get_nested_value(data, key)
                if value is not None:
                    metadata[key] = value
        
        # Extract content
        if self.content_key and isinstance(data, dict):
            content = self._get_nested_value(data, self.content_key)
            if content is None:
                content = ""
            elif not isinstance(content, str):
                content = json.dumps(content, indent=2)
        elif self.extract_all_strings:
            strings = self._extract_all_strings(data)
            content = "\n".join(strings)
        else:
            # Default: format entire JSON as string
            content = json.dumps(data, indent=2)
        
        # Emit structural metadata for downstream JsonEntryChunker
        entries_data = []
        entry_paths = []
        
        if isinstance(data, list):
            # Array of entries — each item is an entry
            for i, item in enumerate(data):
                entry_paths.append(f"[{i}]")
                entries_data.append(item)
        elif isinstance(data, dict):
            # Object — each top-level key is an entry
            for key in data:
                entry_paths.append(key)
                entries_data.append(data[key])
        
        metadata["entry_paths"] = entry_paths
        metadata["entries_data"] = entries_data
        metadata["entry_count"] = len(entries_data)
        
        return content, metadata
    
    def load(
        self,
        source: Union[str, Path, BinaryIO],
        metadata: Optional[Dict[str, Any]] = None
    ) -> ProcessedDocument:
        """Load a JSON file."""
        meta = metadata or {}
        
        if isinstance(source, (str, Path)):
            path = Path(source)
            with open(path, "r", encoding=self.encoding) as f:
                data = json.load(f)
            source_str = str(path.absolute())
        else:
            content_bytes = source.read()
            if isinstance(content_bytes, bytes):
                content_bytes = content_bytes.decode(self.encoding)
            data = json.loads(content_bytes)
            source_str = meta.get("source", "unknown")
        
        content, json_meta = self._process_json(data)
        meta.update(json_meta)
        
        return ProcessedDocument(
            content=content,
            metadata=meta,
            source=source_str,
            file_type=SupportedFileType.JSON
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
                content_str = await f.read()
            data = json.loads(content_str)
            source_str = str(path.absolute())
        else:
            content_bytes = source.read()
            if isinstance(content_bytes, bytes):
                content_bytes = content_bytes.decode(self.encoding)
            data = json.loads(content_bytes)
            source_str = meta.get("source", "unknown")
        
        content, json_meta = self._process_json(data)
        meta.update(json_meta)
        
        return ProcessedDocument(
            content=content,
            metadata=meta,
            source=source_str,
            file_type=SupportedFileType.JSON
        )
