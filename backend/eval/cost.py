"""Token accounting and price per question.

Counts are exact — taken from the provider's count_tokens endpoint rather than
estimated from characters — and cached on disk by sha256(model + text), so
re-running a config costs nothing.

Prices are USD per million tokens, paid tier, from
https://ai.google.dev/gemini-api/docs/pricing (read 2026-09-30). They are not
discoverable through the API, so they are pinned here and must be refreshed by
hand if Google changes them.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .cache import CACHE_DIR

PRICES_USD_PER_MTOK = {
    "gemini-3.5-flash-lite": {"input": 0.30, "output": 2.50},
    "gemini-3.5-flash": {"input": 1.50, "output": 9.00},
}
PRICES_READ_ON = "2026-09-30"


def price(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """USD for one call. None when the model has no pinned price."""
    rate = PRICES_USD_PER_MTOK.get(model)
    if rate is None:
        return None
    return (input_tokens * rate["input"] + output_tokens * rate["output"]) / 1_000_000


class TokenCounter:
    """Exact token counts from the provider, cached on disk."""

    def __init__(self, api_key: str, root: Path = CACHE_DIR / "tokens"):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.api_key = api_key
        self._client = None
        self._memo: dict[str, int] = {}

    @property
    def client(self):
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def count(self, model: str, text: str) -> int:
        if not text:
            return 0
        key = hashlib.sha256(f"{model}\x1e{text}".encode("utf-8")).hexdigest()
        if key in self._memo:
            return self._memo[key]

        path = self.root / key[:2] / f"{key}.json"
        if path.exists():
            value = json.loads(path.read_text(encoding="utf-8"))["tokens"]
        else:
            value = int(self.client.models.count_tokens(model=model, contents=text).total_tokens)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"tokens": value}), encoding="utf-8")
            tmp.replace(path)

        self._memo[key] = value
        return value
