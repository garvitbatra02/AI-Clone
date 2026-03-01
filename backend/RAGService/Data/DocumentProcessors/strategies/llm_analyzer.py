"""
LLM Structure Analyzer

Uses an LLM to analyze document structure (headings, sections, tables)
for PDFs, or topical grouping for TXT files.

Returns a StructuralMap that downstream chunkers consume as advisory
boundaries. Fully optional — if the LLM fails or returns garbage,
the system falls back to rule-based chunking.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ==================== Data Models ====================

@dataclass
class StructuralElement:
    """
    A structural element identified by the LLM.
    
    Attributes:
        start_line: 1-based start line number.
        end_line: 1-based end line number (inclusive).
        element_type: "heading", "body", "table", "list", "code".
        level: Heading level (1-6), None for non-headings.
        title: Heading text or topic label.
        confidence: 1.0 = cross-validated, 0.5 = LLM-only.
    """
    start_line: int
    end_line: int
    element_type: str = "body"
    level: Optional[int] = None
    title: Optional[str] = None
    confidence: float = 1.0


@dataclass
class StructuralMap:
    """
    Structural analysis result from the LLM.
    
    Attributes:
        elements: List of validated structural elements.
        coverage: Fraction of document lines covered (0.0 - 1.0).
        source: "llm" or "rule-based".
    """
    elements: List[StructuralElement] = field(default_factory=list)
    coverage: float = 0.0
    source: str = "llm"


# ==================== Prompt Templates ====================

STRUCTURAL_ANALYSIS_PROMPT = """You are a document structure analyzer. Given the following document text with numbered lines, identify the structural elements.

Return ONLY a JSON array, no other text. Each element must have exactly these fields:
- "start_line": integer (1-based)
- "end_line": integer (1-based, inclusive)
- "type": one of "heading", "body", "table", "list", "code"
- "level": integer 1-6 (only for headings, null otherwise)
- "title": string (only for headings, null otherwise)

Rules:
- Every line must belong to exactly one element
- Consecutive body paragraphs under the same heading should be one element
- Tables (rows with consistent delimiters like |, tabs, or aligned columns) must be separate elements
- Number lines starting from 1

Document ({line_count} lines):
---
{numbered_lines}
---"""

TOPICAL_GROUPING_PROMPT = """You are a text content analyzer. Given the following text with numbered lines, group consecutive lines by topic.

Return ONLY a JSON array, no other text. Each group must have exactly these fields:
- "start_line": integer (1-based, first line of this topic group)
- "end_line": integer (1-based, inclusive, last line of this topic group)
- "type": "body"
- "title": string (a short 2-5 word label describing the topic of this group)

Rules:
- Every line must belong to exactly one group
- Group related content together (e.g. education info, work experience, skills)
- Keep groups reasonably sized (5-50 lines each)
- Provide meaningful, specific topic labels

Text ({line_count} lines):
---
{numbered_lines}
---"""


# ==================== Analyzer ====================

class LLMStructureAnalyzer:
    """
    Analyzes document structure using an LLM.
    
    Two modes:
    - structural: For PDFs — detects headings, tables, sections.
    - topical: For TXT — groups lines by topic.
    
    Uses the existing ChatService LLM infrastructure (Groq/Cerebras/Gemini)
    for inference. Falls back gracefully on any failure.
    
    Example:
        from ChatService.Chat.llm import LLMFactory, LLMProvider
        
        llm = LLMFactory.create(
            provider=LLMProvider.GROQ,
            model="llama-3.1-8b-instant",
            api_keys=["key1"],
            temperature=0.0,
        )
        
        analyzer = LLMStructureAnalyzer(llm=llm)
        result = analyzer.analyze(text, mode="structural")
        
        if result and result.coverage > 0.5:
            # Use the structural map
            ...
    """
    
    # Max tokens to send in one LLM call (~6000 tokens ≈ 24000 chars)
    MAX_CHARS_PER_WINDOW = 24000
    WINDOW_OVERLAP_LINES = 20
    
    # Minimum coverage to consider the map useful
    MIN_COVERAGE_THRESHOLD = 0.5
    
    def __init__(self, llm: Any = None):
        """
        Args:
            llm: A BaseLLM instance from ChatService.Chat.llm.
                 If None, the analyzer is disabled and analyze() returns None.
        """
        self._llm = llm
    
    @property
    def is_available(self) -> bool:
        """Whether LLM analysis is available."""
        return self._llm is not None
    
    def analyze(
        self,
        text: str,
        mode: str = "structural",
    ) -> Optional[StructuralMap]:
        """
        Analyze document structure using the LLM.
        
        Args:
            text: Full document text.
            mode: "structural" (PDF) or "topical" (TXT).
            
        Returns:
            StructuralMap if successful and coverage > threshold, else None.
        """
        if not self.is_available:
            return None
        
        try:
            lines = text.split("\n")
            total_lines = len(lines)
            
            if total_lines == 0:
                return None
            
            # Check if document fits in one window
            if len(text) <= self.MAX_CHARS_PER_WINDOW:
                raw_elements = self._analyze_window(lines, mode)
            else:
                raw_elements = self._analyze_windowed(lines, mode)
            
            if not raw_elements:
                return None
            
            # Validate elements against source text
            validated = self._validate_elements(raw_elements, lines)
            
            if not validated:
                return None
            
            # Calculate coverage
            covered_lines = set()
            for elem in validated:
                for line_no in range(elem.start_line, elem.end_line + 1):
                    covered_lines.add(line_no)
            
            coverage = len(covered_lines) / total_lines if total_lines > 0 else 0.0
            
            if coverage < self.MIN_COVERAGE_THRESHOLD:
                logger.warning(
                    f"LLM structural map coverage too low: {coverage:.2f} "
                    f"(threshold: {self.MIN_COVERAGE_THRESHOLD})"
                )
                return None
            
            return StructuralMap(
                elements=validated,
                coverage=coverage,
                source="llm",
            )
        
        except Exception as e:
            logger.warning(f"LLM structure analysis failed: {e}")
            return None
    
    async def async_analyze(
        self,
        text: str,
        mode: str = "structural",
    ) -> Optional[StructuralMap]:
        """Async version of analyze."""
        if not self.is_available:
            return None
        
        try:
            lines = text.split("\n")
            total_lines = len(lines)
            
            if total_lines == 0:
                return None
            
            if len(text) <= self.MAX_CHARS_PER_WINDOW:
                raw_elements = await self._async_analyze_window(lines, mode)
            else:
                raw_elements = await self._async_analyze_windowed(lines, mode)
            
            if not raw_elements:
                return None
            
            validated = self._validate_elements(raw_elements, lines)
            
            if not validated:
                return None
            
            covered_lines = set()
            for elem in validated:
                for line_no in range(elem.start_line, elem.end_line + 1):
                    covered_lines.add(line_no)
            
            coverage = len(covered_lines) / total_lines if total_lines > 0 else 0.0
            
            if coverage < self.MIN_COVERAGE_THRESHOLD:
                return None
            
            return StructuralMap(
                elements=validated,
                coverage=coverage,
                source="llm",
            )
        
        except Exception as e:
            logger.warning(f"LLM async structure analysis failed: {e}")
            return None
    
    # ==================== LLM Interaction ====================
    
    def _analyze_window(
        self,
        lines: List[str],
        mode: str,
    ) -> List[Dict[str, Any]]:
        """Analyze a single window of lines via LLM."""
        prompt = self._build_prompt(lines, mode)
        response_text = self._call_llm(prompt)
        return self._parse_response(response_text)
    
    async def _async_analyze_window(
        self,
        lines: List[str],
        mode: str,
    ) -> List[Dict[str, Any]]:
        """Async: analyze a single window."""
        prompt = self._build_prompt(lines, mode)
        response_text = await self._async_call_llm(prompt)
        return self._parse_response(response_text)
    
    def _analyze_windowed(
        self,
        lines: List[str],
        mode: str,
    ) -> List[Dict[str, Any]]:
        """Analyze large documents using overlapping windows."""
        all_elements = []
        window_size = self._estimate_window_size(lines)
        
        start = 0
        while start < len(lines):
            end = min(start + window_size, len(lines))
            window_lines = lines[start:end]
            
            prompt = self._build_prompt(window_lines, mode, line_offset=start)
            response_text = self._call_llm(prompt)
            elements = self._parse_response(response_text)
            
            # Adjust line numbers by offset
            for elem in elements:
                elem["start_line"] = elem.get("start_line", 1) + start
                elem["end_line"] = elem.get("end_line", 1) + start
            
            all_elements.extend(elements)
            start = end - self.WINDOW_OVERLAP_LINES
        
        # Deduplicate overlapping elements
        return self._merge_overlapping_elements(all_elements)
    
    async def _async_analyze_windowed(
        self,
        lines: List[str],
        mode: str,
    ) -> List[Dict[str, Any]]:
        """Async: analyze with overlapping windows."""
        all_elements = []
        window_size = self._estimate_window_size(lines)
        
        start = 0
        while start < len(lines):
            end = min(start + window_size, len(lines))
            window_lines = lines[start:end]
            
            prompt = self._build_prompt(window_lines, mode, line_offset=start)
            response_text = await self._async_call_llm(prompt)
            elements = self._parse_response(response_text)
            
            for elem in elements:
                elem["start_line"] = elem.get("start_line", 1) + start
                elem["end_line"] = elem.get("end_line", 1) + start
            
            all_elements.extend(elements)
            start = end - self.WINDOW_OVERLAP_LINES
        
        return self._merge_overlapping_elements(all_elements)
    
    def _build_prompt(
        self,
        lines: List[str],
        mode: str,
        line_offset: int = 0,
    ) -> str:
        """Build the LLM prompt with numbered lines."""
        numbered = []
        for i, line in enumerate(lines, start=1):
            numbered.append(f"{i + line_offset}: {line}")
        
        numbered_text = "\n".join(numbered)
        line_count = len(lines)
        
        template = (
            STRUCTURAL_ANALYSIS_PROMPT if mode == "structural"
            else TOPICAL_GROUPING_PROMPT
        )
        
        return template.format(
            line_count=line_count,
            numbered_lines=numbered_text,
        )
    
    def _call_llm(self, prompt: str) -> str:
        """Make a synchronous LLM call."""
        from ChatService.Chat.session.chat_session import ChatSession
        
        session = ChatSession(
            system_prompt="You are a document structure analyzer. Return only valid JSON."
        )
        session.add_user_message(prompt)
        
        response = self._llm.chat(session)
        return response.content
    
    async def _async_call_llm(self, prompt: str) -> str:
        """Make an asynchronous LLM call."""
        from ChatService.Chat.session.chat_session import ChatSession
        
        session = ChatSession(
            system_prompt="You are a document structure analyzer. Return only valid JSON."
        )
        session.add_user_message(prompt)
        
        response = await self._llm.async_chat(session)
        return response.content
    
    # ==================== Response Parsing ====================
    
    def _parse_response(self, response_text: str) -> List[Dict[str, Any]]:
        """
        Extract JSON array from LLM response.
        
        Four cascading extraction strategies:
        1. Direct json.loads
        2. Extract from ```json ... ``` fence
        3. Find first [ ... last ]
        4. Line-by-line JSON repair
        """
        if not response_text:
            return []
        
        # Try 1: Direct parse
        try:
            result = json.loads(response_text)
            if isinstance(result, list):
                return result
        except (json.JSONDecodeError, TypeError):
            pass
        
        # Try 2: Extract from markdown fence
        fence_match = re.search(
            r"```(?:json)?\s*\n?(.*?)\n?\s*```",
            response_text,
            re.DOTALL,
        )
        if fence_match:
            try:
                result = json.loads(fence_match.group(1))
                if isinstance(result, list):
                    return result
            except (json.JSONDecodeError, TypeError):
                pass
        
        # Try 3: Find first [ ... last ]
        first_bracket = response_text.find("[")
        last_bracket = response_text.rfind("]")
        if first_bracket != -1 and last_bracket > first_bracket:
            try:
                result = json.loads(response_text[first_bracket:last_bracket + 1])
                if isinstance(result, list):
                    return result
            except (json.JSONDecodeError, TypeError):
                pass
        
        # Try 4: Line-by-line repair
        if first_bracket != -1 and last_bracket > first_bracket:
            raw = response_text[first_bracket:last_bracket + 1]
            # Remove trailing commas before ] or }
            cleaned = re.sub(r",\s*([}\]])", r"\1", raw)
            # Fix single quotes to double quotes
            cleaned = cleaned.replace("'", '"')
            try:
                result = json.loads(cleaned)
                if isinstance(result, list):
                    return result
            except (json.JSONDecodeError, TypeError):
                pass
        
        logger.warning("Failed to parse LLM response as JSON array")
        return []
    
    # ==================== Validation ====================
    
    def _validate_elements(
        self,
        raw_elements: List[Dict[str, Any]],
        lines: List[str],
    ) -> List[StructuralElement]:
        """
        Validate each element against the source text.
        
        - Drops elements with missing/invalid required fields
        - Cross-validates heading claims (must be short lines)
        - Cross-validates table claims (must have delimiters)
        - Clamps out-of-range line numbers
        """
        total_lines = len(lines)
        valid = []
        
        VALID_TYPES = {"heading", "body", "table", "list", "code"}
        TYPE_ALIASES = {
            "header": "heading",
            "headers": "heading",
            "paragraph": "body",
            "text": "body",
            "content": "body",
        }
        
        for elem in raw_elements:
            if not isinstance(elem, dict):
                continue
            
            # Required fields
            start_line = elem.get("start_line")
            end_line = elem.get("end_line")
            elem_type = elem.get("type", "body")
            
            if start_line is None or end_line is None:
                continue
            
            # Validate types
            try:
                start_line = int(start_line)
                end_line = int(end_line)
            except (ValueError, TypeError):
                continue
            
            # Clamp to valid range
            start_line = max(1, min(start_line, total_lines))
            end_line = max(start_line, min(end_line, total_lines))
            
            # Swap if reversed
            if start_line > end_line:
                start_line, end_line = end_line, start_line
            
            # Normalize type
            elem_type = str(elem_type).lower().strip()
            elem_type = TYPE_ALIASES.get(elem_type, elem_type)
            if elem_type not in VALID_TYPES:
                elem_type = "body"
            
            level = elem.get("level")
            title = elem.get("title")
            confidence = 1.0
            
            # Cross-validate heading claims
            if elem_type == "heading":
                actual_line = lines[start_line - 1] if start_line <= total_lines else ""
                if len(actual_line.strip()) > 100:
                    # Too long to be a heading — demote
                    elem_type = "body"
                    confidence = 0.5
                else:
                    if level is not None:
                        try:
                            level = int(level)
                            level = max(1, min(level, 6))
                        except (ValueError, TypeError):
                            level = 1
                    else:
                        level = 1
                    
                    if title is None:
                        title = actual_line.strip()
            
            # Cross-validate table claims
            if elem_type == "table":
                table_lines = lines[start_line - 1:end_line]
                has_delimiters = any(
                    "|" in line or "\t" in line
                    for line in table_lines
                )
                if not has_delimiters:
                    elem_type = "body"
                    confidence = 0.5
            
            valid.append(StructuralElement(
                start_line=start_line,
                end_line=end_line,
                element_type=elem_type,
                level=level,
                title=title,
                confidence=confidence,
            ))
        
        return valid
    
    # ==================== Helpers ====================
    
    def _estimate_window_size(self, lines: List[str]) -> int:
        """Estimate how many lines fit in one LLM window."""
        avg_line_len = sum(len(l) for l in lines[:100]) / min(100, len(lines))
        chars_per_line = max(avg_line_len + 10, 20)  # +10 for line numbers
        return max(50, int(self.MAX_CHARS_PER_WINDOW / chars_per_line))
    
    def _merge_overlapping_elements(
        self,
        elements: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Merge elements from overlapping windows.
        
        In the overlap region, prefer elements from the later window
        (which has more context about what follows).
        """
        if not elements:
            return []
        
        # Sort by start_line
        elements.sort(key=lambda e: e.get("start_line", 0))
        
        merged = []
        for elem in elements:
            if not merged:
                merged.append(elem)
                continue
            
            last = merged[-1]
            last_end = last.get("end_line", 0)
            curr_start = elem.get("start_line", 0)
            
            if curr_start <= last_end:
                # Overlap — keep the one with more specific type
                if elem.get("type", "body") != "body":
                    merged[-1] = elem  # Prefer non-body
                # Otherwise keep existing
            else:
                merged.append(elem)
        
        return merged
