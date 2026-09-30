"""Metric arithmetic, checked on hand-built spans rather than on the corpus."""

import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from eval.metrics.chunking import (  # noqa: E402
    chunks_overlapping,
    contains,
    coverage,
    fragmentation,
    merge_spans,
    overlap,
    structural_metrics,
)
from eval.metrics.retrieval import (  # noqa: E402
    aggregate,
    aggregate_by,
    is_hit,
    rank_of_gold,
    retrieval_metrics,
)


def chunk(start, end, text=None, **extra):
    return {"start": start, "end": end,
            "text": text if text is not None else "x" * (end - start),
            "metadata": f"[{start},{end})", **extra}


# ── span algebra ─────────────────────────────────────────────────────

def test_overlap():
    assert overlap((0, 10), (5, 15)) == 5
    assert overlap((0, 10), (10, 20)) == 0      # half-open: touching is not overlapping
    assert overlap((0, 10), (20, 30)) == 0
    assert overlap((0, 100), (40, 50)) == 10


def test_merge_spans_unions_overlaps():
    assert merge_spans([(0, 10), (5, 20), (30, 40)]) == [(0, 20), (30, 40)]
    assert merge_spans([(30, 40), (0, 10)]) == [(0, 10), (30, 40)]
    assert merge_spans([]) == []


def test_contains():
    assert contains((0, 100), (10, 20))
    assert contains((0, 100), (0, 100))
    assert not contains((0, 100), (90, 110))


# ── structural ───────────────────────────────────────────────────────

def test_orphan_ratio_counts_union_not_sum():
    """An overlapping chunker must not be able to claim >100% coverage."""
    raw = "x" * 1000
    overlapping = [chunk(0, 600), chunk(400, 1000)]     # 1200 chars summed, 1000 union
    assert structural_metrics(overlapping, raw)["orphan_ratio"] == 0.0


def test_orphan_ratio_reports_uncovered_text():
    raw = "x" * 1000
    assert structural_metrics([chunk(0, 250)], raw)["orphan_ratio"] == 0.75


def test_structural_metrics_on_empty_chunking():
    m = structural_metrics([], "x" * 100)
    assert m["n_chunks"] == 0 and m["orphan_ratio"] == 1.0


def test_unique_articles_ignores_placeholders():
    chunks = [chunk(0, 10, article="Article 1"),
              chunk(10, 20, article="Article 1"),
              chunk(20, 30, article="N/A"),
              chunk(30, 40)]
    assert structural_metrics(chunks, "x" * 40)["unique_articles"] == 1


# ── coverage / fragmentation ─────────────────────────────────────────

def test_coverage_needs_a_single_containing_chunk():
    evidence = [(100, 200)]
    assert coverage(evidence, [chunk(0, 300)]) == 1.0
    # Split across two chunks: covered in total, but not by any one chunk.
    assert coverage(evidence, [chunk(0, 150), chunk(150, 300)]) == 0.0


def test_coverage_is_a_fraction_of_spans():
    evidence = [(10, 20), (100, 200)]
    assert coverage(evidence, [chunk(0, 50)]) == 0.5


def test_coverage_of_a_question_with_no_evidence_is_one():
    """unanswerable / out_of_scope questions must not be filtered out."""
    assert coverage([], [chunk(0, 10)]) == 1.0


def test_fragmentation_counts_chunks_per_span():
    evidence = [(0, 300)]
    assert fragmentation(evidence, [chunk(0, 300)]) == 1.0
    assert fragmentation(evidence, [chunk(0, 100), chunk(100, 200), chunk(200, 300)]) == 3.0


def test_fragmentation_without_evidence_is_zero_not_one():
    assert fragmentation([], [chunk(0, 10)]) == 0.0


def test_chunks_overlapping_selects_gold_context():
    chunks = [chunk(0, 100), chunk(100, 200), chunk(200, 300)]
    gold = chunks_overlapping([(150, 250)], chunks)
    assert [c["metadata"] for c in gold] == ["[100,200)", "[200,300)"]


# ── retrieval ────────────────────────────────────────────────────────

def test_is_hit_uses_offsets_not_metadata():
    """The recursive chunker has no article labels; overlap must still score."""
    unlabelled = chunk(150, 250)
    assert is_hit(unlabelled, [(100, 200)])
    assert not is_hit(chunk(900, 1000), [(100, 200)])


def test_is_hit_rejects_chunks_without_offsets():
    with pytest.raises(ValueError):
        is_hit({"metadata": "no offsets", "start": None, "end": None}, [(0, 10)])


def test_is_hit_respects_min_overlap():
    assert is_hit(chunk(199, 300), [(100, 200)])                     # 1 char
    assert not is_hit(chunk(199, 300), [(100, 200)], min_overlap=5)


def test_retrieval_metrics_perfect():
    m = retrieval_metrics([chunk(0, 100)], [(10, 20)])
    assert m["hit_at_k"] and m["recall_at_k"] == 1.0
    assert m["precision_at_k"] == 1.0 and m["mrr"] == 1.0


def test_retrieval_metrics_miss():
    m = retrieval_metrics([chunk(0, 100), chunk(100, 200)], [(500, 600)])
    assert m["hit_at_k"] is False
    assert m["recall_at_k"] == 0.0 and m["mrr"] == 0.0


def test_mrr_reflects_position_of_first_hit():
    retrieved = [chunk(0, 100), chunk(100, 200), chunk(200, 300)]
    assert retrieval_metrics(retrieved, [(250, 260)])["mrr"] == pytest.approx(1 / 3)
    assert retrieval_metrics(retrieved, [(150, 160)])["mrr"] == pytest.approx(1 / 2)


def test_precision_penalises_unearned_context():
    retrieved = [chunk(0, 100), chunk(1000, 1100), chunk(2000, 2100)]
    m = retrieval_metrics(retrieved, [(10, 20)])
    assert m["precision_at_k"] == pytest.approx(1 / 3)
    assert m["recall_at_k"] == 1.0


def test_recall_is_over_spans_not_chunks():
    retrieved = [chunk(0, 100)]
    m = retrieval_metrics(retrieved, [(10, 20), (5000, 5010)])
    assert m["recall_at_k"] == 0.5


def test_metrics_are_none_when_there_is_no_evidence():
    """Unanswerable questions must not be scored as perfect retrieval."""
    m = retrieval_metrics([chunk(0, 100)], [])
    assert m["hit_at_k"] is None and m["recall_at_k"] is None and m["mrr"] is None


def test_rank_of_gold():
    ranked = [chunk(i * 100, (i + 1) * 100) for i in range(60)]
    assert rank_of_gold(ranked, [(0, 50)]) == 1
    assert rank_of_gold(ranked, [(450, 460)]) == 5
    assert rank_of_gold(ranked, [(4900, 4910)]) == 50        # last rank still inside depth
    assert rank_of_gold(ranked, [(5000, 5010)]) is None      # rank 51, beyond depth 50
    assert rank_of_gold(ranked, []) is None


# ── aggregation ──────────────────────────────────────────────────────

def test_aggregate_skips_undefined_rows():
    rows = [{"hit_at_k": True}, {"hit_at_k": False}, {"hit_at_k": None}]
    assert aggregate(rows, "hit_at_k") == pytest.approx(0.5)


def test_aggregate_returns_none_when_nothing_is_defined():
    assert aggregate([{"hit_at_k": None}], "hit_at_k") is None


def test_aggregate_by_slices():
    rows = [
        {"qtype": "factual", "hit_at_k": True},
        {"qtype": "factual", "hit_at_k": False},
        {"qtype": "multi_hop", "hit_at_k": True},
    ]
    assert aggregate_by(rows, "hit_at_k", "qtype") == {
        "factual": pytest.approx(0.5), "multi_hop": 1.0,
    }


# ── judge verdict parsing ────────────────────────────────────────────

def test_parse_verdict_plain():
    from eval.metrics.generation import parse_verdict
    assert parse_verdict('{"score": 1, "reasoning": "matches"}') == (1, "matches")


def test_parse_verdict_tolerates_fences_and_prose():
    from eval.metrics.generation import parse_verdict
    raw = 'Here is my verdict:\n```json\n{"score": 0, "reasoning": "wrong figure"}\n```'
    assert parse_verdict(raw) == (0, "wrong figure")


def test_parse_verdict_stops_at_first_object():
    """The judge sometimes emits two verdicts; a greedy {.*} swallowed both."""
    from eval.metrics.generation import parse_verdict
    raw = '{"score": 1, "reasoning": "ok"}{"score": 0, "reasoning": "second"}'
    assert parse_verdict(raw) == (1, "ok")


def test_parse_verdict_rejects_garbage():
    from eval.metrics.generation import parse_verdict
    with pytest.raises(ValueError):
        parse_verdict("no json here")
    with pytest.raises(ValueError):
        parse_verdict('{"score": 7, "reasoning": "out of range"}')
