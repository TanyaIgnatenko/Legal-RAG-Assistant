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
MAX_WAITS = 8


class QuotaExhausted(RuntimeError):
    """The daily request quota is spent; rerun tomorrow, the cache resumes."""


def with_backoff(call: Callable[[], T]) -> T:
    for attempt in range(MAX_WAITS):
        try:
            return call()
        except Exception as e:  # provider errors arrive wrapped by langchain
            msg = str(e)
            if "RESOURCE_EXHAUSTED" not in msg and "429" not in msg:
                raise
            if "PerDay" in msg:
                raise QuotaExhausted(msg[:400]) from e
            delay = re.search(r"retry in ([\d.]+)s", msg)
            time.sleep((float(delay.group(1)) if delay else 10 * (attempt + 1)) + 1)
    raise QuotaExhausted(f"still rate-limited after {MAX_WAITS} waits")
