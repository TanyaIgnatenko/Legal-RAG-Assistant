"""Step 3.2 — chunk metrics.

Two families:

* structural, needing only the raw text and the chunks (`structural_metrics`)
* evidence-relative, needing gold spans (`coverage`, `fragmentation`)

`coverage` is the ceiling for retrieval: an evidence span split across chunk
boundaries can never be retrieved whole, so questions below coverage == 1 are
excluded from retrieval aggregates rather than silently dragging them down.
"""

from __future__ import annotations

from typing import Iterable, Sequence

Span = tuple[int, int]


# ── span helpers ─────────────────────────────────────────────────────

def overlap(a: Span, b: Span) -> int:
    """Length of the intersection of two half-open spans."""
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def merge_spans(spans: Iterable[Span]) -> list[Span]:
    """Union of possibly overlapping spans, as disjoint sorted spans."""
    ordered = sorted(spans)
    merged: list[Span] = []
    for start, end in ordered:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def contains(outer: Span, inner: Span) -> bool:
    return outer[0] <= inner[0] and inner[1] <= outer[1]


def chunk_spans(chunks: Sequence[dict]) -> list[Span]:
    return [(c["start"], c["end"]) for c in chunks]


# ── structural ───────────────────────────────────────────────────────

def structural_metrics(chunks: Sequence[dict], raw_text: str) -> dict:
    """Shape of a chunking, independent of any dataset.

    `orphan_ratio` is the share of source characters no chunk covers. It counts
    the *union* of chunk spans, so an overlapping chunker cannot inflate its
    coverage past 1.0.
    """
    if not chunks:
        return {
            "n_chunks": 0, "unique_articles": 0,
            "p50_chunk_len": 0, "p95_chunk_len": 0, "max_chunk_len": 0,
            "orphan_ratio": 1.0, "total_chars": len(raw_text), "covered_chars": 0,
        }

    lengths = sorted(len(c["text"]) for c in chunks)
    articles = {c.get("article") for c in chunks} - {None, "N/A"}
    covered = sum(end - start for start, end in merge_spans(chunk_spans(chunks)))

    return {
        "n_chunks": len(chunks),
        "unique_articles": len(articles),
        "p50_chunk_len": _percentile(lengths, 0.50),
        "p95_chunk_len": _percentile(lengths, 0.95),
        "max_chunk_len": lengths[-1],
        "orphan_ratio": round(1 - covered / len(raw_text), 4) if raw_text else 1.0,
        "total_chars": len(raw_text),
        "covered_chars": covered,
    }


# ── evidence-relative ────────────────────────────────────────────────

def coverage(evidence_spans: Sequence[Span], chunks: Sequence[dict]) -> float:
    """Share of evidence spans that fit entirely inside a single chunk.

    Returns 1.0 for a question with no evidence spans (unanswerable /
    out_of_scope), since there is nothing that could fail to be covered.
    """
    if not evidence_spans:
        return 1.0
    spans = chunk_spans(chunks)
    whole = sum(1 for ev in evidence_spans if any(contains(c, ev) for c in spans))
    return whole / len(evidence_spans)


def fragmentation(evidence_spans: Sequence[Span], chunks: Sequence[dict]) -> float:
    """Mean number of chunks each evidence span is cut across (1.0 = ideal).

    Returns 0.0 when there are no evidence spans, so it never claims a span was
    fragmented when none existed.
    """
    if not evidence_spans:
        return 0.0
    spans = chunk_spans(chunks)
    counts = [sum(1 for c in spans if overlap(c, ev) > 0) for ev in evidence_spans]
    return sum(counts) / len(counts)


def chunks_overlapping(evidence_spans: Sequence[Span], chunks: Sequence[dict]) -> list[dict]:
    """The gold chunks: every chunk intersecting any evidence span, in order."""
    return [c for c in chunks
            if any(overlap((c["start"], c["end"]), ev) > 0 for ev in evidence_spans)]


def _percentile(ordered: Sequence[int], q: float) -> int:
    if not ordered:
        return 0
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]
