"""
Shared utilities used across ChatService and RAGService.

Provides:
- KeyRotationMixin: Multi-key rotation, client caching, and retry logic
- AllKeysFailedError: Raised when every API key has been exhausted
"""

from shared.key_rotation import KeyRotationMixin, AllKeysFailedError

__all__ = [
    "KeyRotationMixin",
    "AllKeysFailedError",
]
