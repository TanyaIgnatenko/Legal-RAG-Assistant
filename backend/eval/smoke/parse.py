"""Step 3.1 — parse smoke test.

The first rung of the ceiling ladder. If the PDF does not parse into
recognisable structure, every metric below it is measuring noise, so this
runs first and needs neither a dataset nor an LLM.

Usage:
    python -m eval.smoke.parse [path/to.pdf]
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

BACKEND = Path(__file__).resolve().parents[2]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from src.chunker import ARTICLE_RE, CHAPTER_RE  # noqa: E402
from src.parser import PDFParser  # noqa: E402

GDPR_PDF = BACKEND / "example_data" / "gdpr.pdf"

# (field, low, high) — an open bound is None.
THRESHOLDS: tuple[tuple[str, int | None, int | None], ...] = (
    ("chars", 200_000, None),
    ("articles_found", 90, 110),
    ("chapters_found", 10, 12),
    ("max_line_len", None, 199),
)


@dataclass
class Check:
    field: str
    value: int
    low: int | None
    high: int | None

    @property
    def ok(self) -> bool:
        if self.low is not None and self.value < self.low:
            return False
        if self.high is not None and self.value > self.high:
            return False
        return True

    @property
    def bound(self) -> str:
        if self.low is not None and self.high is not None:
            return f"{self.low}..{self.high}"
        if self.low is not None:
            return f">= {self.low}"
        return f"<= {self.high}"


def parse_smoke(text: str) -> dict:
    """Structural facts about a parsed document."""
    lines = text.split("\n")
    line_lengths = [len(line) for line in lines]
    return {
        "chars": len(text),
        "lines": len(lines),
        "articles_found": len(ARTICLE_RE.findall(text)),
        "chapters_found": len(CHAPTER_RE.findall(text)),
        "max_line_len": max(line_lengths) if line_lengths else 0,
        "p50_line_len": _percentile(line_lengths, 0.50),
        "p95_line_len": _percentile(line_lengths, 0.95),
    }


def check(metrics: dict) -> list[Check]:
    return [Check(f, metrics[f], lo, hi) for f, lo, hi in THRESHOLDS]


def _percentile(values: Iterable[int], q: float) -> int:
    ordered = sorted(values)
    if not ordered:
        return 0
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


def main(pdf_path: Path) -> int:
    metrics = parse_smoke(PDFParser.parse(str(pdf_path)))
    checks = check(metrics)

    print(f"parse smoke: {pdf_path.name}")
    for c in checks:
        print(f"  {'OK  ' if c.ok else 'FAIL'}  {c.field:<16} = {c.value:<9,} ({c.bound})")
    for extra in ("lines", "p50_line_len", "p95_line_len"):
        print(f"  ....  {extra:<16} = {metrics[extra]:,}")

    failed = [c.field for c in checks if not c.ok]
    print(f"\n{'PASS' if not failed else 'FAIL: ' + ', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else GDPR_PDF
    sys.exit(main(target))
