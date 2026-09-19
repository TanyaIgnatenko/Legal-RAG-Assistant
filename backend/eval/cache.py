"""Step 3.5 — disk caches for LLM calls and embeddings.

Debugging a runner must not cost a full paid run each time. Every LLM call is
keyed by sha256(model + prompt + temperature); every embedding matrix by
(model_name, sha256(texts)). Both live under eval/.cache/, which is ignored.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

CACHE_DIR = Path(__file__).resolve().parent / ".cache"


def _sha(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class LLMCache:
    """Prompt -> completion, one JSON file per call."""

    def __init__(self, root: Path = CACHE_DIR / "llm"):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(model: str, prompt: str, temperature: float, salt: str = "") -> str:
        # `salt` separates repeated samples of the same prompt (judge runs).
        return _sha(json.dumps([model, prompt, float(temperature), salt]))

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> str | None:
        path = self._path(key)
        if not path.exists():
            self.misses += 1
            return None
        self.hits += 1
        return json.loads(path.read_text(encoding="utf-8"))["value"]

    def set(self, key: str, value: str) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"value": value}, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)

    def stats(self) -> str:
        return f"llm cache: {self.hits} hits, {self.misses} misses"


class EmbeddingCache:
    """(model_name, sha256(texts)) -> float32 matrix, one .npy per batch."""

    def __init__(self, root: Path = CACHE_DIR / "embeddings"):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, model_name: str, texts: Sequence[str]) -> Path:
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name)
        digest = _sha("\x1e".join(texts))[:32]
        return self.root / f"{slug}-{digest}.npy"

    def get_or_compute(
        self,
        model_name: str,
        texts: Sequence[str],
        compute: Callable[[list[str]], list[list[float]]],
    ) -> np.ndarray:
        path = self._path(model_name, texts)
        if path.exists():
            return np.load(path)
        matrix = np.asarray(compute(list(texts)), dtype=np.float32)
        np.save(path, matrix)
        return matrix
