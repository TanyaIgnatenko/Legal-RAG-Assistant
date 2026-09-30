"""Regression gate for CI: does this change make retrieval worse?

    python -m eval.gate            # check against the committed baseline
    python -m eval.gate --update   # record the current numbers as the baseline

Runs the production configuration end to end except for the LLM: parse, chunk
and retrieval only. That keeps the gate free, deterministic and fast enough to
sit on every pull request — the same reason `--fast` exists in the runner.

A metric may not fall below its baseline. Improvements pass and are reported,
but the baseline is only moved by an explicit `--update`, so a change that
quietly trades one metric for another has to be acknowledged in a diff.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from eval.runner import RESULTS, EvalConfig, Workspace, evaluate  # noqa: E402
from eval.smoke.parse import THRESHOLDS, parse_smoke  # noqa: E402
from src.chunker import get_chunker  # noqa: E402
from src.rag_system import DEFAULT_EMBEDDING_MODEL, DEFAULT_TOP_K  # noqa: E402

BASELINE = RESULTS / "baseline.json"
# Floating-point slack only — this is not a tolerance for real regressions.
EPSILON = 1e-9

# Metrics that must never decrease.
GUARDED = ["hit_at_k", "recall_at_k", "mrr", "coverage", "n_chunks", "unique_articles"]


def measure() -> dict:
    """The production configuration's retrieval numbers, no LLM involved."""
    config = EvalConfig("hierarchical", DEFAULT_TOP_K, DEFAULT_EMBEDDING_MODEL)
    ws = Workspace(embedding_model=config.embedding_model)
    traces = evaluate(config, ws, full=False)

    scored = [t for t in traces if t.hit_at_k is not None and t.coverage == 1.0]
    if not scored:
        raise SystemExit("gate: no scoreable questions — the dataset or spans are broken")

    chunks = get_chunker(config.chunker).chunk(ws.raw)
    parse = parse_smoke(ws.raw)
    mean = lambda values: sum(values) / len(values)  # noqa: E731

    return {
        "config": config.name,
        "articles_found": parse["articles_found"],
        "chapters_found": parse["chapters_found"],
        "max_line_len": parse["max_line_len"],
        "n_chunks": len(chunks),
        "unique_articles": len({c.get("article") for c in chunks} - {None, "N/A"}),
        "n_scored": len(scored),
        "coverage": mean([t.coverage for t in scored]),
        "hit_at_k": mean([float(t.hit_at_k) for t in scored]),
        "recall_at_k": mean([t.recall_at_k for t in scored]),
        "mrr": mean([t.mrr for t in scored]),
    }


def check_absolute(current: dict) -> list[str]:
    """Parse smoke thresholds — these are absolute, not relative to a baseline."""
    failures = []
    for field, low, high in THRESHOLDS:
        value = current.get(field)
        if value is None:
            continue
        if (low is not None and value < low) or (high is not None and value > high):
            bound = f"{low}..{high}" if low is not None and high is not None else \
                    (f">= {low}" if low is not None else f"<= {high}")
            failures.append(f"{field} = {value} violates {bound}")
    return failures


def check_regressions(current: dict, baseline: dict) -> tuple[list[str], list[str]]:
    regressions, improvements = [], []
    for field in GUARDED:
        was, now = baseline.get(field), current.get(field)
        if was is None or now is None:
            continue
        if now < was - EPSILON:
            regressions.append(f"{field}: {was:.4g} -> {now:.4g}")
        elif now > was + EPSILON:
            improvements.append(f"{field}: {was:.4g} -> {now:.4g}")
    return regressions, improvements


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true",
                    help="write the current numbers as the new baseline")
    args = ap.parse_args()

    current = measure()
    print(f"gate: {current['config']} on {current['n_scored']} scoreable questions")
    for field in GUARDED:
        print(f"  {field:<16} {current[field]:.4g}")

    if args.update or not BASELINE.exists():
        BASELINE.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
        print(f"\nbaseline written -> {BASELINE.relative_to(BACKEND)}")
        return 0

    failures = check_absolute(current)
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    regressions, improvements = check_regressions(current, baseline)

    if baseline.get("config") != current["config"]:
        print(f"\nnote: baseline was recorded for {baseline.get('config')!r}, "
              f"now running {current['config']!r}")

    for line in improvements:
        print(f"  improved: {line}")

    if failures or regressions:
        print("\nFAIL")
        for line in failures:
            print(f"  absolute: {line}")
        for line in regressions:
            print(f"  regressed: {line}")
        print("\nIf the drop is intended, rerun with --update and commit the baseline.")
        return 1

    print("\nPASS")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
