"""Shared retry helper for LLM API calls."""

from __future__ import annotations

import random
import time
from typing import Callable, TypeVar

from coffeebench.agent import ContextOverflowError

T = TypeVar("T")


# Substrings that mean "try again" rather than "you're broken".
_TRANSIENT_KEYWORDS = (
    "timeout",
    "timed out",
    "rate limit",
    "429",
    "overloaded",
    "busy",
    "internal server error",
    "500",
    "502",
    "503",
    "504",
    "connection",
    "remote disconnected",
    "ssl",
    "handshake",
    "socket",
    "broken pipe",
    "disconnected",
    # httpx / httpcore network glitches
    "remoteprotocolerror",
    "readtimeout",
    "connecttimeout",
    "connecterror",
    "readerror",
    "writeerror",
    "poolerror",
    "protocol error",
    # gemini transient 5xx
    "service unavailable",
    "deadline exceeded",
    # OpenRouter occasionally returns partial / malformed JSON when its
    # upstream provider drops the connection mid-stream — surface as a
    # JSONDecodeError ("Expecting value..."). Retrying generally clears it.
    "jsondecodeerror",
    "expecting value",
)

# Substrings that mean "this prompt won't fit". Provider error messages
# vary across Anthropic / OpenAI / Gemini / OpenRouter — keep the list
# permissive but specific enough not to false-positive on other 400s.
_OVERFLOW_KEYWORDS = (
    "prompt is too long",
    "context length",
    "context_length_exceeded",
    "maximum context length",
    "maximum context window",
    "exceeds the model's",
    "input is too long",
    "too many tokens",
    "token limit",
)


def _is_transient(exc: BaseException) -> bool:
    msg = (str(exc) + " " + type(exc).__name__).lower()
    return any(k in msg for k in _TRANSIENT_KEYWORDS)


def _is_context_overflow(exc: BaseException) -> bool:
    msg = (str(exc) + " " + type(exc).__name__).lower()
    return any(k in msg for k in _OVERFLOW_KEYWORDS)


def call_with_retry(
    fn: Callable[[], T],
    *,
    max_attempts: int = 6,
    base_delay: float = 4.0,
    max_delay: float = 60.0,
    label: str = "api",
) -> T:
    """Run `fn()` and retry on transient API errors with backoff + jitter.

    Total wait across attempts is bounded — at base_delay=4 and 6 attempts
    the worst case is ~120 s before giving up, which is plenty for the
    common rate-limit and timeout cases without making the run hang on
    a real outage.
    """
    last_exc: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except KeyboardInterrupt:
            raise
        except BaseException as exc:  # noqa: BLE001 — explicit transient check below
            last_exc = exc
            if _is_context_overflow(exc):
                raise ContextOverflowError(
                    f"{type(exc).__name__}: {exc!s:.200s}"
                ) from exc
            if not _is_transient(exc) or attempt == max_attempts:
                raise
            wait = min(max_delay, base_delay * (2 ** (attempt - 1)))
            wait += random.uniform(0, base_delay)
            print(
                f"[retry:{label}] attempt {attempt}/{max_attempts} hit transient "
                f"error: {type(exc).__name__}: {exc!s:.120s} — sleeping {wait:.1f}s"
            )
            time.sleep(wait)
    # Unreachable, but keeps type-checkers honest.
    assert last_exc is not None
    raise last_exc
