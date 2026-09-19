"""Step 5 — chunking ablation: hierarchical vs recursive-512-128 × top_k.

    python -m eval.ablations.chunking --fast
    python -m eval.ablations.chunking --full

20 questions is too few for absolute numbers to separate two systems (a
hit-rate CI is roughly ±20pp at n=20). But every configuration answers the
same questions, so the comparison is paired: McNemar's exact test looks only
at the questions on which the two chunkers disagree.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from scipy.stats import binomtest

BACKEND = Path(__file__).resolve().parents[2]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from eval.runner import RESULTS, EvalConfig, Trace, Workspace, run  # noqa: E402

CHUNKERS = ["hierarchical", "recursive-512-128"]
TOP_KS = [1, 3, 5]
SUMMARY = RESULTS / "ablation_chunking.json"

DIAGNOSES = {
    (True, True, True): "does not test RAG — model knows it unaided",
    (True, True, False): "working as intended",
    (True, False, False): "retriever",
    (True, False, True): "retriever (model knew it; context misled)",
    (False, False, False): "generation / prompt",
    (False, True, True): "dataset label suspect",
    (False, True, False): "dataset label suspect",
    (False, False, True): "context hurts generation",
}


def mcnemar(a: list[dict], b: list[dict], field: str) -> dict:
    """Exact McNemar on a boolean field, paired by question id.

    Only questions scored in both configurations count (retrieval fields are
    None for unanswerable items, and both sides must have coverage == 1).
    b = A right / B wrong, c = A wrong / B right.
    """
    by_id = {t["qid"]: t for t in b}
    pairs = [
        (x[field], by_id[x["qid"]][field])
        for x in a
        if x["qid"] in by_id
        and x[field] is not None and by_id[x["qid"]][field] is not None
        and x["coverage"] == 1.0 and by_id[x["qid"]]["coverage"] == 1.0
    ]
    b_only = sum(1 for x, y in pairs if x and not y)
    c_only = sum(1 for x, y in pairs if y and not x)
    discordant = b_only + c_only
    p = binomtest(b_only, discordant, 0.5).pvalue if discordant else 1.0
    return {"field": field, "n_paired": len(pairs), "b": b_only, "c": c_only,
            "discordant": discordant, "p_value": p}


def diagnosis(traces: list[dict]) -> Counter:
    combos = Counter()
    for t in traces:
        key = (t["correct_oracle"], t["correct_retrieved"], t["correct_noctx"])
        if None not in key:
            combos[key] += 1
    return combos


def load(name: str) -> list[dict]:
    path = RESULTS / f"{name}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> int:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--fast", action="store_true")
    mode.add_argument("--full", action="store_true")
    ap.add_argument("--full-top-k", type=int, nargs="*", default=None,
                    help="restrict generation to these k (saves LLM quota)")
    args = ap.parse_args()

    ws = Workspace(api_key=os.getenv("GEMINI_API_KEY", ""))
    for chunker in CHUNKERS:
        for k in TOP_KS:
            full = args.full and (args.full_top_k is None or k in args.full_top_k)
            run(EvalConfig(chunker, k), ws, full=full)

    tests = []
    for k in TOP_KS:
        a, b = load(f"{CHUNKERS[0]}-k{k}"), load(f"{CHUNKERS[1]}-k{k}")
        for field in ("hit_at_k", "correct_retrieved"):
            t = mcnemar(a, b, field)
            if t["n_paired"]:
                tests.append({"top_k": k, "a": CHUNKERS[0], "b": CHUNKERS[1], **t})

    print("\nMcNemar (exact), b = hierarchical-only right, c = recursive-only right")
    for t in tests:
        print(f"  k={t['top_k']} {t['field']:<18} n={t['n_paired']:>2} "
              f"b={t['b']} c={t['c']} p={t['p_value']:.3f}")

    diagnoses = {}
    for chunker in CHUNKERS:
        for k in TOP_KS:
            combos = diagnosis(load(f"{chunker}-k{k}"))
            if combos:
                diagnoses[f"{chunker}-k{k}"] = {
                    f"{int(o)}{int(r)}{int(n)}": {"n": c, "diagnosis": DIAGNOSES[(o, r, n)]}
                    for (o, r, n), c in sorted(combos.items(), reverse=True)
                }

    SUMMARY.write_text(json.dumps(
        {"chunkers": CHUNKERS, "top_ks": TOP_KS, "mcnemar": tests, "diagnoses": diagnoses},
        indent=2) + "\n", encoding="utf-8")
    print(f"\n-> {SUMMARY.relative_to(BACKEND)}")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
