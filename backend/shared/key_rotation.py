"""
Generic API Key Rotation Mixin.

Provides multi-key rotation, automatic env-loading, per-key client
caching, and retry-with-rotation loops that any service can inherit.

Used by BaseLLM (ChatService), BaseEmbeddings, and BaseReranker
(RAGService) so that key rotation logic is written once and reused
everywhere.

Resilience features:
- Automatic API key loading from environment variables
- Multiple API keys with automatic rotation on failure
- Per-key client caching (both sync and async clients)
- Configurable retry on invalid/empty response
- Streaming variants with first-chunk validation

External callers never need to handle, pass, or even know about
API keys — everything is fully encapsulated.
"""

from __future__ import annotations

import logging
import os
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Iterator,
    Optional,
    TypeVar,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ============================================================================
# Exception
# ============================================================================


class AllKeysFailedError(Exception):
    """
    Raised when all API keys for a service have been exhausted.

    Attributes:
        service: The service/provider label (e.g. "groq", "cohere-embed").
        model: The model that was requested.
        errors: Dictionary mapping key indices to their error messages.
    """

    def __init__(self, service: str, model: str, errors: dict[int, str]):
        self.service = service
        # Keep .provider and .model as aliases for backward compat
        self.provider = service
        self.model = model
        self.errors = errors
        key_errors = "; ".join(f"key_{k}: {v}" for k, v in errors.items())
        super().__init__(
            f"All API keys failed for {service} with model {model}. "
            f"Errors: {key_errors}"
        )


# ============================================================================
# Mixin
# ============================================================================


class KeyRotationMixin:
    """
    Mixin providing multi-key rotation, env auto-loading, client caching,
    and generic "execute with rotation" loops.

    Subclasses must define:
        ENV_VAR_NAME  – class-level str, the env var holding comma-separated keys
        _initialize_client(api_key: str) -> None  – create provider client(s)

    _initialize_client is expected to store client objects as instance
    attributes (e.g. self._client, self._async_client).  The mixin
    automatically caches and restores them during key rotation.

    Usage in a base class::

        class BaseEmbeddings(KeyRotationMixin, ABC):
            ENV_VAR_NAME = ""  # overridden by subclasses

            def __init__(self, config):
                self.config = config
                self._init_key_rotation(config.api_keys, config.max_retries)

            def embed_query(self, text):
                return self._execute_with_rotation(
                    operation=lambda: self._do_embed_query(text),
                    is_valid=lambda r: r is not None and len(r) > 0,
                    service_label=self.provider.value,
                    model_label=self.config.model_name,
                )
    """

    ENV_VAR_NAME: str = ""

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_key_rotation(
        self,
        api_keys: Optional[list[str]] = None,
        max_retries: int = 2,
    ) -> None:
        """
        Bootstrap multi-key state.  Call this from the host class's __init__.

        1. Resolves API keys (explicit list → env var → error).
        2. Creates client for the first key and caches it.

        Args:
            api_keys: Explicit list of API keys (auto-loaded from env if None).
            max_retries: Maximum retry sweeps across all keys on invalid response.
        """
        self._api_keys: list[str] = api_keys or self._load_api_keys_from_env()
        self._max_retries: int = max_retries
        self._client_cache: dict[int, dict[str, Any]] = {}

        if not self._api_keys:
            raise ValueError(
                f"No API keys available for {self.__class__.__name__}. "
                f"Set the {self.ENV_VAR_NAME} environment variable with "
                f"comma-separated API keys."
            )

        # Initialise with the first key and cache the resulting clients
        self._initialize_client(self._api_keys[0])  # type: ignore[attr-defined]
        self._cache_clients(0)

    def _load_api_keys_from_env(self) -> list[str]:
        """
        Load API keys from the environment variable ``ENV_VAR_NAME``.

        Reads the variable, splits on ``,``, strips whitespace.
        Subclasses may override to try fallback env var names.

        Returns:
            List of API key strings (may be empty).
        """
        if not self.ENV_VAR_NAME:
            return []
        keys_str = os.getenv(self.ENV_VAR_NAME, "")
        return [k.strip() for k in keys_str.split(",") if k.strip()]

    # ------------------------------------------------------------------
    # Client cache
    # ------------------------------------------------------------------

    def _cache_clients(self, key_index: int) -> None:
        """Snapshot ``_client`` and ``_async_client`` into the cache."""
        self._client_cache[key_index] = {
            "_client": getattr(self, "_client", None),
            "_async_client": getattr(self, "_async_client", None),
        }

    def _restore_clients(self, key_index: int) -> None:
        """Restore cached ``_client`` / ``_async_client`` attributes."""
        cached = self._client_cache[key_index]
        self._client = cached["_client"]
        async_client = cached.get("_async_client")
        if async_client is not None:
            self._async_client = async_client

    def _get_client_for_key(self, key_index: int) -> None:
        """
        Switch the active client(s) to those for *key_index*.

        Creates and caches clients on first access for that index.
        """
        if key_index not in self._client_cache:
            self._initialize_client(self._api_keys[key_index])  # type: ignore[attr-defined]
            self._cache_clients(key_index)
        else:
            self._restore_clients(key_index)

    # ------------------------------------------------------------------
    # Sync rotation executor
    # ------------------------------------------------------------------

    def _execute_with_rotation(
        self,
        operation: Callable[[], T],
        is_valid: Callable[[T], bool],
        service_label: str,
        model_label: str,
    ) -> T:
        """
        Execute *operation* with automatic key rotation and retry.

        Outer loop: ``max_retries + 1`` attempts.
        Inner loop: every API key in order.

        Args:
            operation: Zero-arg callable that performs the actual API call
                       (typically a closure over service-specific args).
            is_valid: Predicate returning True when the result is acceptable.
            service_label: Human-readable service name (for logs / errors).
            model_label: Model identifier (for logs / errors).

        Returns:
            The first valid result.

        Raises:
            AllKeysFailedError: If all keys × retries are exhausted.
        """
        last_errors: dict[int, str] = {}

        for retry_attempt in range(self._max_retries + 1):
            errors: dict[int, str] = {}

            if retry_attempt > 0:
                logger.info(
                    f"Retrying all keys for {service_label} "
                    f"(attempt {retry_attempt + 1}/{self._max_retries + 1})…"
                )

            for key_index in range(len(self._api_keys)):
                try:
                    self._get_client_for_key(key_index)
                    result = operation()

                    if not is_valid(result):
                        msg = "Invalid/empty response"
                        errors[key_index] = msg
                        logger.warning(
                            f"{service_label} key {key_index} returned "
                            f"invalid response (attempt {retry_attempt + 1}). "
                            f"Trying next key…"
                        )
                        continue

                    # Success
                    logger.info(
                        f"Successfully served request using "
                        f"{service_label} key {key_index}"
                    )
                    return result

                except AllKeysFailedError:
                    raise  # never swallow our own sentinel
                except Exception as e:
                    errors[key_index] = str(e)
                    logger.warning(
                        f"{service_label} key {key_index} failed: {e}. "
                        f"Trying next key…"
                    )

            last_errors = errors

            if retry_attempt < self._max_retries:
                logger.warning(
                    f"All keys for {service_label} failed on attempt "
                    f"{retry_attempt + 1}. Retrying…"
                )

        logger.error(
            f"All {len(self._api_keys)} keys failed for {service_label} "
            f"after {self._max_retries + 1} attempts"
        )
        raise AllKeysFailedError(service_label, model_label, last_errors)

    # ------------------------------------------------------------------
    # Async rotation executor
    # ------------------------------------------------------------------

    async def _execute_with_rotation_async(
        self,
        operation: Callable[[], Awaitable[T]],
        is_valid: Callable[[T], bool],
        service_label: str,
        model_label: str,
    ) -> T:
        """Async variant of :meth:`_execute_with_rotation`."""
        last_errors: dict[int, str] = {}

        for retry_attempt in range(self._max_retries + 1):
            errors: dict[int, str] = {}

            if retry_attempt > 0:
                logger.info(
                    f"Retrying all async keys for {service_label} "
                    f"(attempt {retry_attempt + 1}/{self._max_retries + 1})…"
                )

            for key_index in range(len(self._api_keys)):
                try:
                    self._get_client_for_key(key_index)
                    result = await operation()

                    if not is_valid(result):
                        errors[key_index] = "Invalid/empty response"
                        logger.warning(
                            f"{service_label} key {key_index} returned "
                            f"invalid response (attempt {retry_attempt + 1}). "
                            f"Trying next key…"
                        )
                        continue

                    logger.info(
                        f"Successfully served async request using "
                        f"{service_label} key {key_index}"
                    )
                    return result

                except AllKeysFailedError:
                    raise
                except Exception as e:
                    errors[key_index] = str(e)
                    logger.warning(
                        f"{service_label} key {key_index} failed: {e}. "
                        f"Trying next key…"
                    )

            last_errors = errors

            if retry_attempt < self._max_retries:
                logger.warning(
                    f"All async keys for {service_label} failed on attempt "
                    f"{retry_attempt + 1}. Retrying…"
                )

        logger.error(
            f"All {len(self._api_keys)} async keys failed for "
            f"{service_label} after {self._max_retries + 1} attempts"
        )
        raise AllKeysFailedError(service_label, model_label, last_errors)

    # ------------------------------------------------------------------
    # Sync streaming rotation
    # ------------------------------------------------------------------

    def _stream_with_rotation(
        self,
        operation: Callable[[], Iterator[T]],
        service_label: str,
        model_label: str,
    ) -> Iterator[T]:
        """
        Streaming rotation: try each key, validate by pulling the first
        chunk, then yield the rest.  No outer retry — streaming is
        single-pass over keys.
        """
        errors: dict[int, str] = {}

        for key_index in range(len(self._api_keys)):
            try:
                self._get_client_for_key(key_index)
                stream = operation()
                first_chunk = next(stream)

                logger.info(
                    f"Successfully streaming using "
                    f"{service_label} key {key_index}"
                )
                yield first_chunk
                yield from stream
                return

            except AllKeysFailedError:
                raise
            except Exception as e:
                errors[key_index] = str(e)
                logger.warning(
                    f"{service_label} key {key_index} failed: {e}. "
                    f"Trying next key…"
                )

        logger.error(
            f"All {len(self._api_keys)} keys failed for "
            f"streaming {service_label}"
        )
        raise AllKeysFailedError(service_label, model_label, errors)

    # ------------------------------------------------------------------
    # Async streaming rotation
    # ------------------------------------------------------------------

    async def _stream_with_rotation_async(
        self,
        operation: Callable[[], AsyncIterator[T]],
        service_label: str,
        model_label: str,
    ) -> AsyncIterator[T]:
        """Async variant of :meth:`_stream_with_rotation`."""
        errors: dict[int, str] = {}

        for key_index in range(len(self._api_keys)):
            try:
                self._get_client_for_key(key_index)
                stream = operation()
                first_chunk = await stream.__anext__()

                logger.info(
                    f"Successfully async streaming using "
                    f"{service_label} key {key_index}"
                )
                yield first_chunk
                async for chunk in stream:
                    yield chunk
                return

            except AllKeysFailedError:
                raise
            except Exception as e:
                errors[key_index] = str(e)
                logger.warning(
                    f"{service_label} key {key_index} failed: {e}. "
                    f"Trying next key…"
                )

        logger.error(
            f"All {len(self._api_keys)} async keys failed for "
            f"streaming {service_label}"
        )
        raise AllKeysFailedError(service_label, model_label, errors)
