"""McNemar pairing and Cohen's kappa."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.ablations.chunking import mcnemar  # noqa: E402
from eval.judge_validation import cohens_kappa  # noqa: E402
from eval.report import roman  # noqa: E402
from src.rag_system import response_text  # noqa: E402


def t(qid, hit, coverage=1.0):
    return {"qid": qid, "hit_at_k": hit, "coverage": coverage}


def test_mcnemar_counts_only_discordant_pairs():
    a = [t("1", True), t("2", True), t("3", False), t("4", True)]
    b = [t("1", True), t("2", False), t("3", True), t("4", False)]
    r = mcnemar(a, b, "hit_at_k")
    assert (r["n_paired"], r["b"], r["c"], r["discordant"]) == (4, 2, 1, 3)
    assert r["p_value"] == pytest.approx(1.0)


def test_mcnemar_skips_undefined_and_uncovered():
    a = [t("1", None), t("2", True, coverage=0.0), t("3", True)]
    b = [t("1", True), t("2", False), t("3", False)]
    r = mcnemar(a, b, "hit_at_k")
    assert r["n_paired"] == 1 and r["b"] == 1


def test_mcnemar_with_no_disagreement_is_p_one():
    a = [t("1", True)]
    assert mcnemar(a, a, "hit_at_k")["p_value"] == 1.0


def test_mcnemar_exact_p_matches_binomial():
    a = [t(str(i), True) for i in range(6)]
    b = [t(str(i), False) for i in range(6)]
    assert mcnemar(a, b, "hit_at_k")["p_value"] == pytest.approx(2 / 64)


def test_kappa():
    assert cohens_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert cohens_kappa([1, 1, 0, 0], [1, 0, 1, 0]) == pytest.approx(0.0)
    assert cohens_kappa([1, 1, 1, 0], [1, 1, 0, 0]) == pytest.approx(0.5)


def test_roman_sorting():
    assert sorted(["IX", "I", "XI", "IV", "N/A", "V"], key=roman) == \
        ["I", "IV", "V", "IX", "XI", "N/A"]


def test_response_text_flattens_gemini3_blocks():
    blocks = [{"type": "text", "text": "Hello ", "extras": {"signature": "x"}},
              {"type": "text", "text": "world"}]
    assert response_text(blocks) == "Hello world"
    assert response_text("plain") == "plain"


# ── regression gate ──────────────────────────────────────────────────

def _baseline():
    return {"config": "hierarchical-k10-bge-small-en-v1.5", "hit_at_k": 0.8125,
            "recall_at_k": 0.78, "mrr": 0.51, "coverage": 1.0,
            "n_chunks": 99, "unique_articles": 99}


def test_gate_passes_on_identical_numbers():
    from eval.gate import check_regressions
    regressions, improvements = check_regressions(_baseline(), _baseline())
    assert not regressions and not improvements


def test_gate_catches_a_drop():
    from eval.gate import check_regressions
    current = _baseline() | {"hit_at_k": 0.75}
    regressions, _ = check_regressions(current, _baseline())
    assert len(regressions) == 1 and "hit_at_k" in regressions[0]


def test_gate_reports_improvement_without_failing():
    from eval.gate import check_regressions
    current = _baseline() | {"hit_at_k": 0.875}
    regressions, improvements = check_regressions(current, _baseline())
    assert not regressions and len(improvements) == 1


def test_gate_catches_a_broken_chunker():
    """A chunker that stops finding articles must not slip through."""
    from eval.gate import check_regressions
    current = _baseline() | {"n_chunks": 26, "unique_articles": 26}
    regressions, _ = check_regressions(current, _baseline())
    assert len(regressions) == 2


def test_gate_absolute_thresholds_catch_a_broken_parse():
    from eval.gate import check_absolute
    assert not check_absolute({"chars": 360_853, "articles_found": 99,
                               "chapters_found": 11, "max_line_len": 138})
    failures = check_absolute({"chars": 360_853, "articles_found": 12,
                               "chapters_found": 11, "max_line_len": 900})
    assert len(failures) == 2
