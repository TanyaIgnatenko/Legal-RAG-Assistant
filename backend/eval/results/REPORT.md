# Evaluation report — chunking ablation

Dataset `gdpr_qa_resolved.json` (sha `88cb15aade247218`), 20 questions. Ground truth: `quote_spans` (the sentence that answers the question, located verbatim in the PDF).
Embedding model `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. Chunker code sha `6d9b0289c9c90b05`. Generated 2026-09-20T09:13:59+00:00.

## 1. Ceiling ladder

Read top to bottom and fix the **first** rung that breaks, not the lowest number.

**hierarchical, k=3**

```
parse      articles=99 chapters=11 max_line=138         OK
chunk      n=99 orphan=45% coverage=1.00 frag=1.00      OK
retrieve   hit@3=0.38 (on coverage=1, n=16) MRR=0.28    <- median gold rank 5, 2 not in top-50
generate   (withheld: judge not validated against human labels)
```

**recursive-512-128, k=3**

```
parse      articles=99 chapters=11 max_line=138         OK
chunk      n=1012 orphan=0% coverage=1.00 frag=1.41     OK
retrieve   hit@3=0.12 (on coverage=1, n=16) MRR=0.09    <- median gold rank 23, 8 not in top-50
generate   (withheld: judge not validated against human labels)
```

## 2. Ablation

| config | chunks | orphan | cov | frag | hit@k | recall@k | prec@k | MRR | med. gold rank | ctx chars |
|---|---|---|---|---|---|---|---|---|---|---|
| hierarchical-k1 | 99 | 45% | 1.00 | 1.00 | 0.19 | 0.16 | 0.19 | 0.19 | 5 | 2,340 |
| hierarchical-k3 | 99 | 45% | 1.00 | 1.00 | 0.38 | 0.38 | 0.15 | 0.28 | 5 | 5,900 |
| hierarchical-k5 | 99 | 45% | 1.00 | 1.00 | 0.44 | 0.44 | 0.10 | 0.30 | 5 | 10,305 |
| recursive-512-128-k1 | 1,012 | 0% | 1.00 | 1.41 | 0.06 | 0.06 | 0.06 | 0.06 | 23 | 459 |
| recursive-512-128-k3 | 1,012 | 0% | 1.00 | 1.41 | 0.12 | 0.12 | 0.04 | 0.09 | 23 | 1,374 |
| recursive-512-128-k5 | 1,012 | 0% | 1.00 | 1.41 | 0.12 | 0.12 | 0.03 | 0.09 | 23 | 2,273 |

Retrieval metrics are averaged over answerable questions with coverage = 1 only.

**McNemar's exact test** (paired on the same questions; only questions where the two chunkers disagree carry information):

| k | metric | paired n | hier. only | recur. only | exact p |
|---|---|---|---|---|---|
| 1 | hit_at_k | 16 | 3 | 1 | 0.625 |
| 3 | hit_at_k | 16 | 5 | 1 | 0.219 |
| 5 | hit_at_k | 16 | 6 | 1 | 0.125 |

## 3. Slices (k=3)

### hit@3 by question type

| qtype | n | hierarchical | recursive-512-128 |
|---|---|---|---|
| factual | 12 | 0.42 | 0.17 |
| multi_hop | 4 | 0.25 | 0.00 |

### hit@3 by chapter

| chapter | n | hierarchical | recursive-512-128 |
|---|---|---|---|
| I | 1 | 0.00 | 0.00 |
| II | 2 | 0.50 | 0.50 |
| III | 2 | 0.00 | 0.00 |
| IV | 3 | 0.67 | 0.00 |
| V | 1 | 1.00 | 0.00 |
| VI | 1 | 0.00 | 1.00 |
| VII | 1 | 0.00 | 0.00 |
| VIII | 2 | 0.50 | 0.00 |
| IX | 1 | 1.00 | 0.00 |
| X | 1 | 0.00 | 0.00 |
| XI | 1 | 0.00 | 0.00 |

## 4. Failure diagnosis

Pattern of (correct with oracle context, with retrieved context, with no context).

_Withheld until the judge is validated (see §5)._

## 5. Judge validation

Not yet validated. Generation metrics are withheld until `python -m eval.judge_validation score` reports kappa ≥ 0.6.

## 6. Limitations

> 20 questions, stratified across GDPR chapters and 4 question types. This is not a statistically
> representative sample — absolute metrics have wide confidence intervals (±~20pp at n=20).
> Configurations are compared pairwise on the same question set; the ablation table reports
> McNemar's exact test on that paired comparison. Generation metrics use Gemini as a judge on
> Gemini-generated answers; judge–human agreement is reported separately.

Further, specific to this setup:

- The two chunkers do not receive the same context budget at equal k: an article averages ~2,000 characters, a recursive chunk ~450. hit@k for the hierarchical chunker therefore rewards retrieving a much larger target.
- The judge (`gemini-3.5-flash-lite`) is the same model that generated the answers, because the free-tier quota for a stronger judge is 20 requests a day. Models tend to rate their own output favourably, which is one more reason the generation numbers are only published if judge-human kappa reaches 0.6.
- The hierarchical chunker does not index the 173 recitals (the preamble before Chapter I, 45% of the text). No question in this dataset is answered only by a recital, so that gap is invisible here rather than harmless.
