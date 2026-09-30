"""Rate-limit handling for eval-time LLM calls.

The production path swallows LLM errors into an "Error generating answer"
string. In an evaluation that string would be judged like a real answer, so
eval calls go through here instead: per-minute limits are waited out, a daily
quota stops the run (everything finished so far is already in the cache).
"""

from __future__ import annotations

import re
import time
from typing import Callable, TypeVar

T = TypeVar("T")
MAX_WAITS = 10


class QuotaExhausted(RuntimeError):
    """The daily request quota is spent; rerun tomorrow, the cache resumes."""


def with_backoff(call: Callable[[], T]) -> T:
    for attempt in range(MAX_WAITS):
        try:
            return call()
        except Exception as e:  # provider errors arrive wrapped by langchain
            msg = str(e)
            rate_limited = "RESOURCE_EXHAUSTED" in msg or "429" in msg
            # 503/500 mean the model is busy, not that anything is wrong with
            # the request; they are as worth waiting out as a rate limit.
            transient = any(code in msg for code in ("UNAVAILABLE", "503", "INTERNAL", "500"))
            if not rate_limited and not transient:
                raise
            if "PerDay" in msg:
                raise QuotaExhausted(msg[:400]) from e
            delay = re.search(r"retry in ([\d.]+)s", msg)
            time.sleep((float(delay.group(1)) if delay else 10 * (attempt + 1)) + 1)
    raise QuotaExhausted(f"still rate-limited after {MAX_WAITS} waits")
