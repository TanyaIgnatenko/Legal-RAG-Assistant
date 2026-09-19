"""Step 6 — render eval/results/REPORT.md from the ablation traces.

    python -m eval.report
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import pandas as pd

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from eval.ablations.chunking import CHUNKERS, SUMMARY, TOP_KS, load  # noqa: E402
from eval.judge_validation import JSON_PATH as JUDGE_JSON  # noqa: E402
from eval.runner import RESULTS  # noqa: E402

REPORT = RESULTS / "REPORT.md"
HEADLINE_K = 3

LIMITATIONS = (
    "> 20 questions, stratified across GDPR chapters and 4 question types. This is not a statistically\n"
    "> representative sample — absolute metrics have wide confidence intervals (±~20pp at n=20).\n"
    "> Configurations are compared pairwise on the same question set; the ablation table reports\n"
    "> McNemar's exact test on that paired comparison. Generation metrics use Gemini as a judge on\n"
    "> Gemini-generated answers; judge–human agreement is reported separately."
)


def meta(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.meta.json").read_text(encoding="utf-8"))


def frame(name: str) -> pd.DataFrame:
    return pd.DataFrame(load(name))


def mean(series: pd.Series) -> float | None:
    s = series.dropna().astype(float)
    return float(s.mean()) if len(s) else None


def fmt(x, pct=False, digits=2) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    return f"{x:.0%}" if pct else f"{x:.{digits}f}"


def judge_status() -> dict | None:
    return json.loads(JUDGE_JSON.read_text(encoding="utf-8")) if JUDGE_JSON.exists() else None


def has_generation(df: pd.DataFrame) -> bool:
    return "correct_retrieved" in df and df["correct_retrieved"].notna().any()


def row_metrics(name: str) -> dict:
    df, m = frame(name), meta(name)
    answerable = df[df["hit_at_k"].notna()]
    covered = answerable[answerable["coverage"] == 1.0]
    ranks = [r for r in answerable["rank_of_gold"] if pd.notna(r)]
    out = {
        "config": name,
        "n_chunks": m["chunks"]["n_chunks"],
        "orphan": m["chunks"]["orphan_ratio"],
        "coverage": mean(answerable["coverage"]),
        "fragmentation": mean(answerable["fragmentation"]),
        "n_scored": len(covered),
        "hit": mean(covered["hit_at_k"]),
        "recall": mean(covered["recall_at_k"]),
        "precision": mean(covered["precision_at_k"]),
        "mrr": mean(covered["mrr"]),
        "median_rank": statistics.median(ranks) if ranks else None,
        "not_in_top50": int(answerable["rank_of_gold"].isna().sum()),
        "ctx_chars": mean(df["n_context_chars"]),
    }
    if has_generation(df):
        unans = df[df["qtype"].isin(["unanswerable", "out_of_scope"])]
        ans = df[~df["qtype"].isin(["unanswerable", "out_of_scope"])]
        out |= {
            "oracle": mean(df["correct_oracle"]),
            "retrieved": mean(df["correct_retrieved"]),
            "noctx": mean(df["correct_noctx"]),
            "faithful": mean(df["faithful"]),
            "citation_ok": mean(ans["citation_ok"]),
            "abstention_unans": mean(unans["abstained"]),
            "false_abstention": mean(ans["abstained"]),
        }
    return out


def ladder(name: str, publish_generation: bool) -> list[str]:
    m, r = meta(name), row_metrics(name)
    p, c = m["parse"], m["chunks"]
    parse_ok = 90 <= p["articles_found"] <= 110 and 10 <= p["chapters_found"] <= 12
    lines = [
        f"parse      articles={p['articles_found']} chapters={p['chapters_found']} "
        f"max_line={p['max_line_len']}".ljust(56) + ("OK" if parse_ok else "<- BREAK"),
        f"chunk      n={c['n_chunks']} orphan={c['orphan_ratio']:.0%} "
        f"coverage={fmt(r['coverage'])} frag={fmt(r['fragmentation'])}".ljust(56)
        + ("OK" if r["coverage"] == 1.0 else f"<- {1 - r['coverage']:.0%} of evidence split"),
        f"retrieve   hit@{HEADLINE_K}={fmt(r['hit'])} (on coverage=1, n={r['n_scored']}) "
        f"MRR={fmt(r['mrr'])}".ljust(56)
        + (f"<- median gold rank {fmt(r['median_rank'], digits=0)}, "
           f"{r['not_in_top50']} not in top-50" if (r["hit"] or 0) < 0.8 else "OK"),
    ]
    if "retrieved" in r and publish_generation:
        lines.append(f"generate   oracle={fmt(r['oracle'])} retrieved={fmt(r['retrieved'])} "
                     f"noctx={fmt(r['noctx'])}")
    elif "retrieved" in r:
        lines.append("generate   (withheld: judge not validated against human labels)")
    else:
        lines.append("generate   (not run: --fast)")
    return lines


def ablation_table(publish_generation: bool) -> list[str]:
    rows = [row_metrics(f"{c}-k{k}") for c in CHUNKERS for k in TOP_KS]
    cols = [("config", "config", None), ("n_chunks", "chunks", "int"),
            ("orphan", "orphan", "pct"), ("coverage", "cov", None),
            ("fragmentation", "frag", None), ("hit", "hit@k", None),
            ("recall", "recall@k", None), ("precision", "prec@k", None),
            ("mrr", "MRR", None), ("median_rank", "med. gold rank", "int"),
            ("ctx_chars", "ctx chars", "int")]
    if publish_generation and any("retrieved" in r for r in rows):
        cols += [("oracle", "correct oracle", None), ("retrieved", "correct retr.", None),
                 ("noctx", "correct noctx", None), ("faithful", "faithful", None)]

    def cell(r, key, kind):
        v = r.get(key)
        if kind == "int":
            return "—" if v is None else f"{v:,.0f}"
        if kind == "pct":
            return fmt(v, pct=True)
        return v if key == "config" else fmt(v)

    out = ["| " + " | ".join(h for _, h, _ in cols) + " |",
           "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(cell(r, k, t)) for k, _, t in cols) + " |" for r in rows]
    return out


def mcnemar_lines(publish_generation: bool) -> list[str]:
    tests = json.loads(SUMMARY.read_text(encoding="utf-8"))["mcnemar"]
    out = ["| k | metric | paired n | hier. only | recur. only | exact p |",
           "|---|---|---|---|---|---|"]
    for t in tests:
        if t["field"] == "correct_retrieved" and not publish_generation:
            continue
        out.append(f"| {t['top_k']} | {t['field']} | {t['n_paired']} | {t['b']} | "
                   f"{t['c']} | {t['p_value']:.3f} |")
    return out


def roman(value: str) -> int:
    """Sort key for chapter numerals; non-numerals sort last."""
    digits = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
    if not value or any(ch not in digits for ch in value):
        return 10_000
    total = 0
    for ch, nxt in zip(value, value[1:] + " "):
        total += -digits[ch] if digits.get(nxt, 0) > digits[ch] else digits[ch]
    return total


def slices(field: str, key: str) -> list[str]:
    frames = {c: frame(f"{c}-k{HEADLINE_K}") for c in CHUNKERS}
    keys = sorted({v for df in frames.values() for v in df[key]},
                  key=roman if key == "chapter" else str)
    out = [f"| {key} | n | " + " | ".join(CHUNKERS) + " |", "|---|---|" + "---|" * len(CHUNKERS)]
    for value in keys:
        cells, n = [], 0
        for c in CHUNKERS:
            df = frames[c]
            sub = df[(df[key] == value) & df[field].notna()]
            if "coverage" in sub and field.startswith(("hit", "recall", "mrr")):
                sub = sub[sub["coverage"] == 1.0]
            n = len(sub)
            cells.append(fmt(mean(sub[field])))
        if n:
            out.append(f"| {value} | {n} | " + " | ".join(cells) + " |")
    return out


def diagnosis_lines() -> list[str]:
    diag = json.loads(SUMMARY.read_text(encoding="utf-8")).get("diagnoses", {})
    if not diag:
        return ["_Not available: generation was not run._"]
    out = ["| config | oracle | retrieved | noctx | n | diagnosis |", "|---|---|---|---|---|---|"]
    for config, combos in diag.items():
        for code, v in combos.items():
            marks = ["✅" if ch == "1" else "❌" for ch in code]
            out.append(f"| {config} | {' | '.join(marks)} | {v['n']} | {v['diagnosis']} |")
    return out


def main() -> int:
    judge = judge_status()
    publish = bool(judge and judge["trusted"])
    lines = ["# Evaluation report — chunking ablation", ""]

    first = meta(f"{CHUNKERS[0]}-k{HEADLINE_K}")
    lines += [
        f"Dataset `{first['dataset_file']}` (sha `{first['dataset_hash']}`), "
        f"{first['n_questions']} questions. Ground truth: `{first['ground_truth']}` "
        "(the sentence that answers the question, located verbatim in the PDF).",
        f"Embedding model `{first['config']['embedding_model']}`. "
        f"Chunker code sha `{first['chunker_code_hash']}`. Generated {first['date']}.",
        "",
        "## 1. Ceiling ladder",
        "",
        "Read top to bottom and fix the **first** rung that breaks, not the lowest number.",
        "",
    ]
    for c in CHUNKERS:
        lines += [f"**{c}, k={HEADLINE_K}**", "", "```", *ladder(f"{c}-k{HEADLINE_K}", publish), "```", ""]

    lines += ["## 2. Ablation", "", *ablation_table(publish), "",
              "Retrieval metrics are averaged over answerable questions with coverage = 1 only.",
              "",
              "**McNemar's exact test** (paired on the same questions; "
              "only questions where the two chunkers disagree carry information):", "",
              *mcnemar_lines(publish), ""]

    lines += ["## 3. Slices (k=3)", "", "### hit@3 by question type", "",
              *slices("hit_at_k", "qtype"), "", "### hit@3 by chapter", "",
              *slices("hit_at_k", "chapter"), ""]

    lines += ["## 4. Failure diagnosis", "",
              "Pattern of (correct with oracle context, with retrieved context, with no context).", ""]
    lines += diagnosis_lines() if publish else [
        "_Withheld until the judge is validated (see §5)._"]
    lines += [""]

    lines += ["## 5. Judge validation", ""]
    if judge:
        lines += [f"Cohen's kappa between the Gemini judge and a human on {judge['n']} random "
                  f"answers: **{judge['cohens_kappa']}** (raw agreement {judge['agreement']:.0%}). "
                  + ("Above the 0.6 bar — generation metrics are reported."
                     if judge["trusted"] else
                     "Below the 0.6 bar — the judge is not trusted and generation metrics are withheld."),
                  ""]
    else:
        lines += ["Not yet validated. Generation metrics are withheld until "
                  "`python -m eval.judge_validation score` reports kappa ≥ 0.6.", ""]

    lines += ["## 6. Limitations", "", LIMITATIONS, "",
              "Further, specific to this setup:", "",
              "- The two chunkers do not receive the same context budget at equal k: an article "
              "averages ~2,000 characters, a recursive chunk ~450. hit@k for the hierarchical "
              "chunker therefore rewards retrieving a much larger target.",
              "- The hierarchical chunker does not index the 173 recitals (the preamble before "
              "Chapter I, 45% of the text). No question in this dataset is answered only by a "
              "recital, so that gap is invisible here rather than harmless.",
              ""]

    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"-> {REPORT.relative_to(BACKEND)}")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
