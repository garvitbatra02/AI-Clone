"""
Centralised prompt registry for the entire ChatClone application.

Every system prompt, context template, and instruction string used
anywhere in the codebase lives here.  Import what you need:

    from prompts import RAG_SYSTEM_PROMPT, CONTEXT_HEADER
    from prompts.rag import RAG_SYSTEM_PROMPT
"""

from .rag import (
    RAG_SYSTEM_PROMPT,
    CONTEXT_HEADER,
    CONTEXT_CHUNK_TEMPLATE,
    CONTEXT_SEPARATOR,
)

__all__ = [
    # RAG prompts
    "RAG_SYSTEM_PROMPT",
    "CONTEXT_HEADER",
    "CONTEXT_CHUNK_TEMPLATE",
    "CONTEXT_SEPARATOR",
]
