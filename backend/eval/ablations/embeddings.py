"""Embedding ablation: does the retrieval ceiling come from the embedding model?

    python -m eval.ablations.embeddings --fast
    python -m eval.ablations.embeddings --full --full-top-k 3

The deployed system embeds an English-only regulation with a multilingual
model. The chunking ablation showed retrieval is the bottleneck, and
`rank_of_gold` showed the gold passage often sits far down the list rather
than just below the cutoff — which points at the encoder, not at top_k.

Chunker is held fixed at hierarchical so the only thing that varies is the
embedding model.

Each configuration runs in its own process: several sentence-transformers
models cannot be loaded into one interpreter here. This module therefore
imports nothing heavy at module scope — a parent already holding torch
leaves the child without the memory to load its own copy.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

CHUNKER = "hierarchical"
TOP_KS = [1, 3, 5, 7, 10]
BASELINE = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODELS = [
    BASELINE,                                    # deployed today
    "sentence-transformers/all-MiniLM-L6-v2",    # English, tiny
    "BAAI/bge-small-en-v1.5",                    # English, retrieval-tuned
]
RESULTS = BACKEND / "eval" / "results"
SUMMARY = RESULTS / "ablation_embeddings.json"


def config_name(model: str, k: int) -> str:
    """Mirrors EvalConfig.name without importing it (see module docstring)."""
    if model == BASELINE:
        return f"{CHUNKER}-k{k}"
    return f"{CHUNKER}-k{k}-{model.split('/')[-1]}"


def run_in_subprocess(model: str, k: int, full: bool, attempts: int = 3) -> None:
    cmd = [sys.executable, "-m", "eval.runner",
           "--chunker", CHUNKER, "--top-k", str(k),
           "--embedding-model", model, "--full" if full else "--fast"]
    # torch and faiss each ship their own OpenMP runtime; without this the
    # child aborts on Windows while loading the encoder.
    env = {**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE", "OMP_NUM_THREADS": "1",
           "HF_HUB_DISABLE_PROGRESS_BARS": "1"}

    for attempt in range(attempts):
        proc = subprocess.run(cmd, cwd=BACKEND, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", env=env)
        if proc.returncode == 0:
            for line in (proc.stdout or "").splitlines():
                if "hit@" in line or "llm cache" in line:
                    print(line)
            return
        # Windows has not yet reclaimed the previous encoder's pages.
        if "paging file is too small" in (proc.stderr or "") and attempt + 1 < attempts:
            print(f"  {config_name(model, k)}: out of paging space, retrying")
            time.sleep(15)
            continue
        break

    tail = "\n".join((proc.stderr or "").splitlines()[-6:])
    raise SystemExit(
        f"{config_name(model, k)} failed (exit {proc.returncode}):\n{tail}")


def compare(models: list[str]) -> list[dict]:
    """Paired McNemar of each candidate against the deployed model."""
    from eval.ablations.chunking import load, mcnemar   # heavy; import late

    tests = []
    for model in models[1:]:
        for k in TOP_KS:
            a, b = load(config_name(models[0], k)), load(config_name(model, k))
            for field in ("hit_at_k", "correct_retrieved"):
                t = mcnemar(a, b, field)
                if t["n_paired"]:
                    tests.append({"top_k": k, "baseline": models[0], "model": model, **t})
    return tests


def main() -> int:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--fast", action="store_true")
    mode.add_argument("--full", action="store_true")
    ap.add_argument("--full-top-k", type=int, nargs="*", default=None,
                    help="restrict generation to these k (saves LLM quota)")
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--compare-only", action="store_true",
                    help="skip the runs, just re-read existing traces")
    args = ap.parse_args()

    if not args.compare_only:
        for model in args.models:
            for k in TOP_KS:
                full = args.full and (args.full_top_k is None or k in args.full_top_k)
                run_in_subprocess(model, k, full)

    tests = compare(args.models)
    print("\nMcNemar (exact), b = baseline only right, c = candidate only right")
    for t in tests:
        print(f"  k={t['top_k']} {t['field']:<18} {t['model'].split('/')[-1]:<24} "
              f"n={t['n_paired']:>2} b={t['b']} c={t['c']} p={t['p_value']:.3f}")

    SUMMARY.write_text(json.dumps(
        {"chunker": CHUNKER, "models": args.models, "top_ks": TOP_KS, "mcnemar": tests},
        indent=2) + "\n", encoding="utf-8")
    print(f"\n-> {SUMMARY.relative_to(BACKEND)}")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
