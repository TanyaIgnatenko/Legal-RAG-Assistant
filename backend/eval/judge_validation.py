"""Judge validation: does the LLM judge agree with a human?

    python -m eval.judge_validation export [--config recursive-512-128-k3]
        -> eval/results/judge_validation.csv with 15 random (answer, verdict)
           pairs and an empty `human_score` column to fill with 0 / 1

    python -m eval.judge_validation score
        -> Cohen's kappa between human_score and judge_score,
           written to eval/results/judge_validation.json

If kappa < 0.6 the judge is not trusted and generation metrics stay out of
the README.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from eval.runner import RESULTS  # noqa: E402

CSV_PATH = RESULTS / "judge_validation.csv"
JSON_PATH = RESULTS / "judge_validation.json"
N_PAIRS = 15
SEED = 20260919
KAPPA_THRESHOLD = 0.6
FIELDS = ["pair_id", "config", "qid", "variant", "question", "reference_answer",
          "answer", "judge_score", "judge_reasoning", "human_score"]


def cohens_kappa(a: list[int], b: list[int]) -> float:
    """Cohen's kappa for two binary raters."""
    n = len(a)
    if n == 0:
        raise ValueError("no ratings")
    observed = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1 - expected)


def export(configs: list[str]) -> int:
    dataset = {q["id"]: q for q in json.loads(
        (BACKEND / "eval" / "dataset" / "gdpr_qa_resolved.json").read_text(encoding="utf-8"))}

    pool = []
    for config in configs:
        path = RESULTS / f"{config}.jsonl"
        if not path.exists():
            print(f"missing {path.name}; run the full eval first")
            return 1
        for line in path.read_text(encoding="utf-8").splitlines():
            t = json.loads(line)
            for variant in ("oracle", "retrieved", "noctx"):
                if t.get(f"correct_{variant}") is None:
                    continue
                pool.append({
                    "config": config, "qid": t["qid"], "variant": variant,
                    "question": dataset[t["qid"]]["question"],
                    "reference_answer": dataset[t["qid"]]["reference_answer"],
                    "answer": t[f"a_{variant}"],
                    "judge_score": int(t[f"correct_{variant}"]),
                    "judge_reasoning": t["judge_reasoning"].get(f"correct_{variant}", ""),
                })

    if len(pool) < N_PAIRS:
        print(f"only {len(pool)} judged answers; need {N_PAIRS}")
        return 1

    sample = random.Random(SEED).sample(pool, N_PAIRS)
    with CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for i, row in enumerate(sample, 1):
            # The human labels blind: judge columns are kept but should be
            # hidden while labelling.
            w.writerow({"pair_id": i, **row, "human_score": ""})
    print(f"wrote {N_PAIRS} pairs -> {CSV_PATH.relative_to(BACKEND)}")
    print("fill human_score with 0/1 without looking at judge_score, then run: score")
    return 0


def score() -> int:
    with CSV_PATH.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    labelled = [r for r in rows if r["human_score"].strip() in ("0", "1")]
    if len(labelled) < len(rows):
        print(f"{len(rows) - len(labelled)} rows lack a 0/1 human_score")
        return 1

    human = [int(r["human_score"]) for r in labelled]
    judge = [int(r["judge_score"]) for r in labelled]
    kappa = cohens_kappa(human, judge)
    result = {
        "n": len(labelled),
        "agreement": sum(h == j for h, j in zip(human, judge)) / len(labelled),
        "cohens_kappa": round(kappa, 3),
        "threshold": KAPPA_THRESHOLD,
        "trusted": kappa >= KAPPA_THRESHOLD,
        "disagreements": [r["pair_id"] for r in labelled if r["human_score"] != r["judge_score"]],
    }
    JSON_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("export")
    ex.add_argument("--config", nargs="+",
                    default=["hierarchical-k3", "recursive-512-128-k3"])
    sub.add_parser("score")
    args = ap.parse_args()
    return export(args.config) if args.cmd == "export" else score()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
