# Legal RAG Assistant

AI-powered legal document analysis to help you understand legal documents more easily and accurately.

[![Live Demo](https://img.shields.io/badge/deployed%20on-Vercel-black?style=flat&logo=vercel)](https://legal-rag-demo.vercel.app/)

### 📸 Project Preview

![Main Screen](screenshots/main-screen.png)

### Chat Interface

![Chat Screen](screenshots/chat.png)

## 🏗️ Architecture

```
PDF Document
    ↓
Parse & Chunk (PyMuPDF + Hierarchical chunking)
    ↓
Generate Embeddings (bge-small-en-v1.5)
    ↓
Store in Vector DB (FAISS)
    ↓
User Query → Semantic Search → Retrieve Top-10 Chunks
    ↓
LLM (Gemini) + Context → Generate Answer
```

## 📊 Evaluation

A span-based eval harness lives in [`backend/eval/`](backend/eval). Ground truth is the
sentence that answers each question, located verbatim in the PDF, so retrieval is scored by
character-offset overlap rather than by comparing metadata labels — which is what lets
different chunkers and encoders be compared in one coordinate space. 20 questions across all
11 GDPR chapters (12 factual, 4 multi-hop, 2 unanswerable, 2 out-of-scope).

**The production configuration was chosen by measurement**: hierarchical (article-level)
chunking, `BAAI/bge-small-en-v1.5`, `top_k=10`. Against the previous default —
`paraphrase-multilingual-MiniLM-L12-v2` at `top_k=3` — it wins on every question that
separates the two, and this is the one comparison in the project that reaches significance:

| | previous default | shipped |
|---|---|---|
| hit@k (answering passage retrieved) | 0.38 | **0.81** |
| answers correct | 9/20 | **15/20** |
| false abstentions on answerable questions | 50% | **19%** |
| faithfulness / citation accuracy | 1.00 / 1.00 | 1.00 / 1.00 |
| input tokens per question | 1,516 | 5,582 |
| USD per 1,000 questions | $0.62 | $1.92 |

McNemar's exact test, paired on the same questions: hit@k 7–0 (p = 0.016), answer
correctness 6–0 (p = 0.031). Nothing regressed — no hallucination appeared with the larger
context, and the extra cost is under two dollars per thousand questions.

**Ceiling ladder.** Read top down and fix the first rung that breaks:

```
parse      99 articles, 11 chapters, max line 138       OK
chunk      n=99  coverage=1.00  frag=1.00  orphan=45%   OK (orphan = the recitals)
retrieve   hit@10=0.81  MRR=0.52  median gold rank 2    <- still the ceiling
generate   with oracle context 18/20; with no context 2/20
```

The ladder is what identified the encoder. Retrieval, not generation, was the bottleneck:
the model answers 18/20 when handed the right passage and 2/20 with no context, so every
lost question was a retrieval miss. `rank_of_gold` then separated "k is too small" from
"the encoder never surfaces it" — the old multilingual encoder left 2 of 16 questions
outside the top 50 at any depth, the new one leaves none.

**Ablations** (`backend/eval/ablations/`, both re-runnable):

*Chunking* — hierarchical vs recursive-512-128, k ∈ {1,3,5}. Hierarchical retrieves the
answer more often at every k (hit@3 0.38 vs 0.12) and wins answer correctness 6–0 at k=5
(p = 0.031). Part of its edge is that an article chunk is a ~4× larger target than a
512-character window, which is why the fixed-window chunker is not simply "worse".

*Embedding model* — three encoders, k ∈ {1,3,5,7,10,15,20}:

| encoder | @1 | @3 | @5 | @10 | @20 | reachable in top-50 |
|---|---|---|---|---|---|---|
| paraphrase-multilingual-MiniLM-L12-v2 | 0.19 | 0.38 | 0.44 | 0.69 | 0.75 | 0.88 |
| all-MiniLM-L6-v2 | 0.31 | 0.62 | 0.69 | 0.69 | 0.81 | 0.94 |
| bge-small-en-v1.5 | 0.31 | 0.69 | 0.75 | 0.81 | 0.81 | **1.00** |

The multilingual encoder needs k=20 to reach what bge reaches at k=5 — same hit rate, four
times the context. Its last column is the reason a reranker alone would not have fixed this:
a reranker can only reorder what the encoder already surfaced.

**Not published yet.** Generation metrics above are graded by an LLM judge that has not been
validated against human labels. 15 answers are exported to
`backend/eval/results/judge_validation.csv`; Cohen's κ must reach 0.6 before the report
publishes generation numbers. The judge is the same model that wrote the answers, so this
gate is not a formality.

Full report, slices by question type and chapter, and limitations:
[`backend/eval/results/REPORT.md`](backend/eval/results/REPORT.md).

```bash
cd backend
pip install -r requirements-dev.txt
python -m eval.ablations.chunking --fast     # retrieval only, no LLM, ~1 min
python -m eval.ablations.embeddings --fast   # encoder sweep, no LLM
python -m eval.ablations.chunking --full     # plus generation and judge (needs GEMINI_API_KEY)
python -m eval.report

# after labelling human_score in eval/results/judge_validation.csv
python -m eval.judge_validation score
```

## 🔧 Tech Stack

**Backend:**
- Python, FastAPI
- PyMuPDF, Sentence Transformers, FAISS
- Google Gemini API
- Deployed on Railway

**LLM**: Google Gemini API

**Frontend:**
- Next.js, React, TypeScript
- Tailwind CSS
- Deployed on Vercel

## 💡 Usage

1. Upload a PDF document (e.g., GDPR) or use pre-loaded one
2. Ask questions in natural language
3. Get AI answers with source citations

**Example questions:**
- What is personal data according to GDPR?
- What are the penalties for violations?
- What are data subject rights?

## 🎥 Demo

**Live:** [Try it here](https://legal-rag-demo.vercel.app/)

## 📝 License

MIT

## 👤 Author

**Tatyana Ignatenko**  
[GitHub](https://github.com/TanyaIgnatenko) • [LinkedIn](https://www.linkedin.com/in/tatyana-ignatenko/)
