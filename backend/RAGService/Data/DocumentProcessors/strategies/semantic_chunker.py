"""
Semantic Chunker

Structure-aware chunking for PDF and DOCX documents.
Splits on heading/section boundaries, isolates tables as standalone chunks,
and falls back to RecursiveTextSplitter for oversized sections.

Can optionally consume an LLM-generated StructuralMap for enhanced
boundary detection (especially useful for PDFs where headings are
not detectable from the text alone).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from RAGService.Data.DocumentProcessors.base import (
    ProcessedDocument,
    RecursiveTextSplitter,
    TextSplitterConfig,
)
from RAGService.Data.DocumentProcessors.strategies.base_strategy import (
    BaseChunkingStrategy,
)
from RAGService.Data.VectorDB.base import DocumentChunk


@dataclass
class Section:
    """A section of a document defined by heading boundaries."""
    title: Optional[str] = None
    level: int = 0
    start_char: int = 0
    end_char: int = 0
    content: str = ""
    is_table: bool = False
    children: List["Section"] = field(default_factory=list)


class SemanticChunker(BaseChunkingStrategy):
    """
    Structure-aware chunker for PDF and DOCX files.
    
    Strategy:
    1. If a heading_hierarchy is available (DOCX), build a section tree
       from heading boundaries.
    2. If an LLM StructuralMap is provided, use it to define sections.
    3. For PDFs without LLM analysis, fall back to paragraph-based
       splitting with page boundary tracking.
    4. Tables (tagged with [TABLE]...[/TABLE]) are always isolated
       as standalone chunks.
    5. Oversized sections are split using RecursiveTextSplitter.
    
    Each chunk gets metadata: heading_path, section_title, page_number,
    has_table, chunk_index, total_chunks.
    """
    
    # Regex for [TABLE]...[/TABLE] blocks
    TABLE_PATTERN = re.compile(
        r"\[TABLE\]\n(.*?)\n\[/TABLE\]",
        re.DOTALL
    )
    
    def chunk(
        self,
        document: ProcessedDocument,
        structural_map: Optional[Any] = None,
    ) -> List[DocumentChunk]:
        """
        Chunk a PDF or DOCX document.
        
        Args:
            document: Loaded document with content and metadata.
            structural_map: Optional LLM StructuralMap for enhanced splitting.
        """
        content = document.content
        metadata = document.metadata
        
        # Determine chunking approach based on available metadata
        heading_hierarchy = metadata.get("heading_hierarchy")
        page_boundaries = metadata.get("page_boundaries")
        
        if structural_map is not None:
            # LLM-assisted: use structural map
            sections = self._sections_from_structural_map(content, structural_map)
        elif heading_hierarchy:
            # DOCX with heading styles detected
            sections = self._sections_from_headings(content, heading_hierarchy)
        else:
            # PDF without LLM — paragraph-based splitting
            sections = self._sections_from_paragraphs(content)
        
        # Build chunks from sections
        chunks = []
        for section in sections:
            section_chunks = self._chunk_section(
                section, document, page_boundaries
            )
            chunks.extend(section_chunks)
        
        # Post-pass: stamp total_chunks and re-index
        for i, chunk in enumerate(chunks):
            chunk.chunk_index = i
            chunk.metadata["chunk_index"] = i
        self._stamp_total_chunks(chunks)
        
        return chunks
    
    # ==================== Section Building ====================
    
    def _sections_from_headings(
        self,
        content: str,
        heading_hierarchy: List[Dict[str, Any]],
    ) -> List[Section]:
        """
        Build flat sections from DOCX heading_hierarchy metadata.
        
        Each heading defines a section boundary. Text between two
        headings belongs to the first heading's section.
        """
        if not heading_hierarchy:
            return [Section(content=content, end_char=len(content))]
        
        sections = []
        
        for i, heading in enumerate(heading_hierarchy):
            start = heading["start_char"]
            # End is the start of next heading, or document end
            if i + 1 < len(heading_hierarchy):
                end = heading_hierarchy[i + 1]["start_char"]
            else:
                end = len(content)
            
            section_text = content[start:end].strip()
            if not section_text:
                continue
            
            # Check for table blocks within this section
            table_sections, text_sections = self._extract_tables_from_text(
                section_text, start
            )
            
            for ts in text_sections:
                sections.append(Section(
                    title=heading.get("title"),
                    level=heading.get("level", 1),
                    start_char=ts[0],
                    end_char=ts[1],
                    content=ts[2],
                    is_table=False,
                ))
            
            for tb in table_sections:
                sections.append(Section(
                    title=heading.get("title"),
                    level=heading.get("level", 1),
                    start_char=tb[0],
                    end_char=tb[1],
                    content=tb[2],
                    is_table=True,
                ))
        
        # Handle content before the first heading
        first_heading_start = heading_hierarchy[0]["start_char"]
        if first_heading_start > 0:
            pre_text = content[:first_heading_start].strip()
            if pre_text:
                sections.insert(0, Section(
                    content=pre_text,
                    end_char=first_heading_start,
                ))
        
        # Sort by start position
        sections.sort(key=lambda s: s.start_char)
        
        return sections
    
    def _sections_from_paragraphs(self, content: str) -> List[Section]:
        """
        Split content into sections by paragraph boundaries.
        
        Used for PDFs without heading metadata. Splits on double
        newlines, then groups consecutive paragraphs into sections
        up to max_chunk_size. Tables are isolated.
        """
        sections = []
        
        # First extract tables
        table_sections, remaining_parts = self._extract_tables_from_text(
            content, 0
        )
        
        for tb in table_sections:
            sections.append(Section(
                start_char=tb[0],
                end_char=tb[1],
                content=tb[2],
                is_table=True,
            ))
        
        # Split remaining text on paragraph boundaries
        for start, end, text in remaining_parts:
            paragraphs = text.split("\n\n")
            current_text = ""
            current_start = start
            
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                
                if current_text and len(current_text) + len(para) + 2 > self.max_chunk_size:
                    # Flush current section
                    sections.append(Section(
                        content=current_text,
                        start_char=current_start,
                        end_char=current_start + len(current_text),
                    ))
                    current_start = current_start + len(current_text) + 2
                    current_text = para
                else:
                    if current_text:
                        current_text += "\n\n" + para
                    else:
                        current_text = para
            
            if current_text:
                sections.append(Section(
                    content=current_text,
                    start_char=current_start,
                    end_char=current_start + len(current_text),
                ))
        
        sections.sort(key=lambda s: s.start_char)
        return sections
    
    def _sections_from_structural_map(
        self,
        content: str,
        structural_map: Any,
    ) -> List[Section]:
        """
        Build sections from an LLM-generated StructuralMap.
        
        The structural_map provides line-based element boundaries.
        We convert line numbers to character offsets and build sections.
        """
        lines = content.split("\n")
        # Build line-number to char-offset mapping
        line_offsets = []
        offset = 0
        for line in lines:
            line_offsets.append(offset)
            offset += len(line) + 1  # +1 for newline
        
        sections = []
        elements = getattr(structural_map, "elements", [])
        
        current_heading_title = None
        current_heading_level = 0
        current_heading_path = []
        
        for element in elements:
            start_line = getattr(element, "start_line", 1) - 1  # 0-indexed
            end_line = getattr(element, "end_line", 1)  # inclusive
            elem_type = getattr(element, "element_type", "body")
            
            # Clamp to valid range
            start_line = max(0, min(start_line, len(lines) - 1))
            end_line = max(1, min(end_line, len(lines)))
            
            start_char = line_offsets[start_line] if start_line < len(line_offsets) else 0
            end_char = (
                line_offsets[end_line] if end_line < len(line_offsets)
                else len(content)
            )
            
            section_text = content[start_char:end_char].strip()
            if not section_text:
                continue
            
            if elem_type == "heading":
                current_heading_title = getattr(element, "title", section_text)
                current_heading_level = getattr(element, "level", 1)
                # Update heading path
                while (
                    current_heading_path
                    and current_heading_path[-1][0] >= current_heading_level
                ):
                    current_heading_path.pop()
                current_heading_path.append(
                    (current_heading_level, current_heading_title)
                )
                continue  # Heading text is included in the next body section
            
            is_table = elem_type == "table"
            
            sections.append(Section(
                title=current_heading_title,
                level=current_heading_level,
                start_char=start_char,
                end_char=end_char,
                content=section_text,
                is_table=is_table,
            ))
        
        if not sections:
            # Fallback: treat entire content as one section
            return [Section(content=content, end_char=len(content))]
        
        return sections
    
    # ==================== Chunk Building ====================
    
    def _chunk_section(
        self,
        section: Section,
        document: ProcessedDocument,
        page_boundaries: Optional[List[Tuple[int, int]]] = None,
    ) -> List[DocumentChunk]:
        """
        Convert a section into one or more DocumentChunks.
        
        - Tables → single chunk with has_table=True
        - Small sections → single chunk
        - Large sections → split with RecursiveTextSplitter
        """
        # Determine heading path for metadata
        heading_path = self._get_heading_path(section, document)
        page_number = self._get_page_number(section.start_char, page_boundaries)
        
        if section.is_table:
            # Table always becomes one chunk
            meta = self._build_base_metadata(
                document,
                section.content,
                chunk_index=0,
                section_title=section.title,
                heading_path=heading_path,
                page_number=page_number,
                has_table=True,
            )
            return [self._make_chunk(
                content=section.content,
                metadata=meta,
                chunk_index=0,
                source=document.source,
                start_char=section.start_char,
                end_char=section.end_char,
            )]
        
        if len(section.content) <= self.max_chunk_size:
            # Section fits in one chunk
            meta = self._build_base_metadata(
                document,
                section.content,
                chunk_index=0,
                section_title=section.title,
                heading_path=heading_path,
                page_number=page_number,
            )
            return [self._make_chunk(
                content=section.content,
                metadata=meta,
                chunk_index=0,
                source=document.source,
                start_char=section.start_char,
                end_char=section.end_char,
            )]
        
        # Section is too large — split with RecursiveTextSplitter
        splitter = RecursiveTextSplitter(TextSplitterConfig(
            chunk_size=self.max_chunk_size,
            chunk_overlap=self.chunk_overlap,
        ))
        text_chunks = splitter.split_text(section.content)
        
        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            # Calculate approximate char position
            sub_start = section.content.find(chunk_text)
            if sub_start >= 0:
                abs_start = section.start_char + sub_start
                abs_end = abs_start + len(chunk_text)
            else:
                abs_start = section.start_char
                abs_end = section.end_char
            
            sub_page = self._get_page_number(abs_start, page_boundaries)
            
            meta = self._build_base_metadata(
                document,
                chunk_text,
                chunk_index=i,
                section_title=section.title,
                heading_path=heading_path,
                page_number=sub_page or page_number,
            )
            chunks.append(self._make_chunk(
                content=chunk_text,
                metadata=meta,
                chunk_index=i,
                source=document.source,
                start_char=abs_start,
                end_char=abs_end,
            ))
        
        return chunks
    
    # ==================== Helpers ====================
    
    def _extract_tables_from_text(
        self,
        text: str,
        base_offset: int,
    ) -> Tuple[List[Tuple[int, int, str]], List[Tuple[int, int, str]]]:
        """
        Extract [TABLE]...[/TABLE] blocks from text.
        
        Returns:
            (table_sections, text_sections) — each as (start, end, content) tuples.
        """
        tables = []
        text_parts = []
        
        last_end = 0
        for match in self.TABLE_PATTERN.finditer(text):
            # Text before table
            if match.start() > last_end:
                pre = text[last_end:match.start()].strip()
                if pre:
                    text_parts.append((
                        base_offset + last_end,
                        base_offset + match.start(),
                        pre,
                    ))
            
            # Table content (without tags)
            table_content = match.group(1).strip()
            tables.append((
                base_offset + match.start(),
                base_offset + match.end(),
                table_content,
            ))
            last_end = match.end()
        
        # Text after last table
        if last_end < len(text):
            remaining = text[last_end:].strip()
            if remaining:
                text_parts.append((
                    base_offset + last_end,
                    base_offset + len(text),
                    remaining,
                ))
        
        # If no tables found, return all text as one part
        if not tables and not text_parts:
            if text.strip():
                text_parts.append((base_offset, base_offset + len(text), text.strip()))
        
        return tables, text_parts
    
    def _get_heading_path(
        self,
        section: Section,
        document: ProcessedDocument,
    ) -> Optional[List[str]]:
        """Build the heading_path for a section."""
        heading_hierarchy = document.metadata.get("heading_hierarchy", [])
        if not heading_hierarchy or section.title is None:
            if section.title:
                return [section.title]
            return None
        
        # Find all headings that are ancestors of this section
        path = []
        for heading in heading_hierarchy:
            if heading["start_char"] > section.start_char:
                break
            if heading["level"] <= section.level:
                # Remove deeper or equal headings from path
                while path and path[-1][0] >= heading["level"]:
                    path.pop()
                path.append((heading["level"], heading["title"]))
        
        if not path:
            if section.title:
                return [section.title]
            return None
        
        return [title for _, title in path]
    
    def _get_page_number(
        self,
        char_offset: int,
        page_boundaries: Optional[List[Tuple[int, int]]],
    ) -> Optional[int]:
        """Determine the page number for a given character offset."""
        if not page_boundaries:
            return None
        
        for i, (start, end) in enumerate(page_boundaries):
            if start <= char_offset < end:
                return i + 1  # 1-based page number
        
        # If offset is past all boundaries, return last page
        if page_boundaries and char_offset >= page_boundaries[-1][0]:
            return len(page_boundaries)
        
        return None
