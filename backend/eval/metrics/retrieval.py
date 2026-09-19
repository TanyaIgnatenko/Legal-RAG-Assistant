"""Step 3.3 — retrieval metrics.

Pure arithmetic: no LLM, no network. A retrieved chunk counts as a hit when its
char span *intersects* a gold evidence span. Matching on metadata instead would
make the recursive chunker unscoreable — its chunks carry no article labels at
all — and would quietly compare the two strategies in different label spaces.
"""

from __future__ import annotations

from typing import Sequence

from .chunking import Span, overlap

DIAGNOSTIC_DEPTH = 50


def is_hit(chunk: dict, evidence_spans: Sequence[Span], min_overlap: int = 1) -> bool:
    """Does this chunk intersect any evidence span by at least `min_overlap` chars?"""
    if chunk.get("start") is None or chunk.get("end") is None:
        raise ValueError(f"chunk {chunk.get('metadata')!r} has no offsets")
    span = (chunk["start"], chunk["end"])
    return any(overlap(span, ev) >= min_overlap for ev in evidence_spans)


def retrieval_metrics(
    retrieved: Sequence[dict],
    evidence_spans: Sequence[Span],
    min_overlap: int = 1,
) -> dict:
    """hit@k / recall@k / precision@k / MRR for one question.

    `recall@k` is over evidence spans (how much of the gold evidence was
    reached), `precision@k` over retrieved chunks (how much of the context
    was earned). With no evidence spans the question is not retrievable by
    construction, and every field comes back as None rather than a
    flattering 1.0.
    """
    k = len(retrieved)
    if not evidence_spans:
        return {"hit_at_k": None, "recall_at_k": None, "precision_at_k": None,
                "mrr": None, "k": k, "n_relevant": 0}

    hits = [is_hit(c, evidence_spans, min_overlap) for c in retrieved]
    spans_found = sum(
        1 for ev in evidence_spans
        if any(overlap((c["start"], c["end"]), ev) >= min_overlap for c in retrieved)
    )
    first = next((i for i, h in enumerate(hits) if h), None)

    return {
        "hit_at_k": any(hits),
        "recall_at_k": spans_found / len(evidence_spans),
        "precision_at_k": (sum(hits) / k) if k else 0.0,
        "mrr": (1.0 / (first + 1)) if first is not None else 0.0,
        "k": k,
        "n_relevant": sum(hits),
    }


def rank_of_gold(
    ranked: Sequence[dict],
    evidence_spans: Sequence[Span],
    min_overlap: int = 1,
) -> int | None:
    """1-based rank of the first relevant chunk in a deep candidate list.

    The key diagnostic number. A rank of 4-10 while k=3 means the retriever
    works and k is simply too small; a rank of 30+ means the embeddings are not
    placing the right text anywhere near the question.
    """
    if not evidence_spans:
        return None
    for i, chunk in enumerate(ranked[:DIAGNOSTIC_DEPTH], start=1):
        if is_hit(chunk, evidence_spans, min_overlap):
            return i
    return None


def aggregate(rows: Sequence[dict], field: str) -> float | None:
    """Mean of `field` over rows where it is defined (None and NaN excluded)."""
    values = [r[field] for r in rows if r.get(field) is not None]
    if not values:
        return None
    return sum(float(v) for v in values) / len(values)


def aggregate_by(rows: Sequence[dict], field: str, key: str) -> dict[str, float | None]:
    """Same mean, sliced by `key` — an average hides one badly-parsed chapter."""
    buckets: dict[str, list[dict]] = {}
    for row in rows:
        buckets.setdefault(str(row.get(key)), []).append(row)
    return {name: aggregate(group, field) for name, group in sorted(buckets.items())}
