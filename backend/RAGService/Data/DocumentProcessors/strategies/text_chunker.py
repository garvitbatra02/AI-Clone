"""
Text Chunker

Chunking strategy for plain text (.txt) files.

Default path: RecursiveTextSplitter with paragraph separators.
LLM-assisted path: Uses topic group boundaries from an LLM StructuralMap,
where each group of related lines becomes one chunk with a 'topic' label.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from RAGService.Data.DocumentProcessors.base import (
    ProcessedDocument,
    RecursiveTextSplitter,
    TextSplitterConfig,
)
from RAGService.Data.DocumentProcessors.strategies.base_strategy import (
    BaseChunkingStrategy,
)
from RAGService.Data.VectorDB.base import DocumentChunk


class TextChunker(BaseChunkingStrategy):
    """
    Chunker for plain text files.
    
    Without LLM analysis:
        Splits on paragraph boundaries (\\n\\n) then sentences,
        using RecursiveTextSplitter. No topic or section metadata.
    
    With LLM StructuralMap:
        The map provides topic groups — line ranges with topic labels.
        Each group becomes a chunk (split further if oversized).
        Chunks get a 'topic' metadata field.
    """
    
    def __init__(
        self,
        max_chunk_size: int = 800,
        chunk_overlap: int = 150,
    ):
        super().__init__(max_chunk_size, chunk_overlap)
    
    def chunk(
        self,
        document: ProcessedDocument,
        structural_map: Optional[Any] = None,
    ) -> List[DocumentChunk]:
        """
        Chunk a plain text document.
        
        Args:
            document: The loaded TXT document.
            structural_map: Optional LLM topical grouping map.
        """
        if structural_map is not None:
            chunks = self._chunk_with_topics(document, structural_map)
        else:
            chunks = self._chunk_recursive(document)
        
        # Post-pass
        for i, chunk in enumerate(chunks):
            chunk.chunk_index = i
            chunk.metadata["chunk_index"] = i
        self._stamp_total_chunks(chunks)
        
        return chunks
    
    def _chunk_recursive(self, document: ProcessedDocument) -> List[DocumentChunk]:
        """
        Default path: split using RecursiveTextSplitter.
        """
        splitter = RecursiveTextSplitter(TextSplitterConfig(
            chunk_size=self.max_chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        ))
        
        text_chunks = splitter.split_text(document.content)
        
        chunks = []
        current_pos = 0
        
        for i, chunk_text in enumerate(text_chunks):
            start = document.content.find(chunk_text, current_pos)
            if start == -1:
                start = current_pos
            end = start + len(chunk_text)
            current_pos = start + 1
            
            meta = self._build_base_metadata(
                document,
                chunk_text,
                chunk_index=i,
            )
            chunks.append(self._make_chunk(
                content=chunk_text,
                metadata=meta,
                chunk_index=i,
                source=document.source,
                start_char=start,
                end_char=end,
            ))
        
        return chunks
    
    def _chunk_with_topics(
        self,
        document: ProcessedDocument,
        structural_map: Any,
    ) -> List[DocumentChunk]:
        """
        LLM-assisted path: use topic group boundaries.
        
        The structural_map has elements with:
            - start_line, end_line: line range for the group
            - element_type: "body" (or "topic_group")
            - title: the topic label (e.g. "Education", "Technical Skills")
        """
        content = document.content
        lines = content.split("\n")
        
        # Build line offset mapping
        line_offsets = []
        offset = 0
        for line in lines:
            line_offsets.append(offset)
            offset += len(line) + 1
        
        elements = getattr(structural_map, "elements", [])
        
        if not elements:
            # Fallback to recursive splitting
            return self._chunk_recursive(document)
        
        chunks = []
        
        for element in elements:
            start_line = max(0, getattr(element, "start_line", 1) - 1)
            end_line = min(len(lines), getattr(element, "end_line", 1))
            topic = getattr(element, "title", None)
            
            # Extract text for this group
            group_lines = lines[start_line:end_line]
            group_text = "\n".join(group_lines).strip()
            
            if not group_text:
                continue
            
            start_char = line_offsets[start_line] if start_line < len(line_offsets) else 0
            end_char = (
                line_offsets[end_line] if end_line < len(line_offsets)
                else len(content)
            )
            
            if len(group_text) <= self.max_chunk_size:
                # Fits in one chunk
                meta = self._build_base_metadata(
                    document,
                    group_text,
                    chunk_index=len(chunks),
                    topic=topic,
                )
                chunks.append(self._make_chunk(
                    content=group_text,
                    metadata=meta,
                    chunk_index=len(chunks),
                    source=document.source,
                    start_char=start_char,
                    end_char=end_char,
                ))
            else:
                # Too large — sub-split but keep topic label
                splitter = RecursiveTextSplitter(TextSplitterConfig(
                    chunk_size=self.max_chunk_size,
                    chunk_overlap=self.chunk_overlap,
                ))
                sub_chunks = splitter.split_text(group_text)
                
                for sub_text in sub_chunks:
                    meta = self._build_base_metadata(
                        document,
                        sub_text,
                        chunk_index=len(chunks),
                        topic=topic,
                    )
                    chunks.append(self._make_chunk(
                        content=sub_text,
                        metadata=meta,
                        chunk_index=len(chunks),
                        source=document.source,
                        start_char=start_char,
                        end_char=end_char,
                    ))
        
        return chunks
