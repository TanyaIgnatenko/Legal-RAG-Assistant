# Legal RAG Assistant

AI-powered legal document analysis to help you understand legal documents more easily and accurately.

[![Live Demo](https://img.shields.io/badge/deployed%20on-Vercel-black?style=flat&logo=vercel)](https://legal-rag-demo.vercel.app/)

## ✨ Features

**Answers you can check**

- 📎 **Inline citations** - every answer cites the GDPR articles it used, verified against
  the retrieved text (100% accurate).
- 📚 **Sources shown** - the passages behind each answer are listed under it.
- 🤐 **Says "I don't know"** - declines instead of guessing. Zero hallucinations measured.

**Quality is measured**

- 📊 **Eval harness** - 20 GDPR questions, each labelled with the sentence that answers it.
- 🔬 **Three ablation studies** - chunking, embedding model, `top_k`. The winner shipped.
- 💶 **Cost per question tracked** - exact tokens and USD next to every quality metric.
- 🚦 **CI regression gate** - each pull request re-runs retrieval and fails on a drop. Free,
  no LLM calls.

**Hardened**

- 🛡️ **Prompt-injection defence** - 12 attacks, including ones hidden in the documents. All
  held.
- 🧾 **No invented law** - fake articles and fake fines planted in the text were rejected.

**Practical**

- 📄 **Bring your own PDF** - upload a contract, or use the pre-loaded GDPR.
- ⚡ **Fast cold start** - the GDPR index is precomputed, so the API boots ready.

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

Eval harness in [`backend/eval/`](backend/eval). 20 GDPR questions across all 11 chapters,
each labelled with the exact sentence that answers it. Retrieval is scored by character
offsets, so different chunkers and encoders are comparable.

Three ablation studies were run, each on the same questions:

1. **Chunking** — article-level (hierarchical) vs fixed 512-character windows with overlap.
2. **Embedding model** — multilingual MiniLM vs all-MiniLM-L6-v2 vs bge-small-en-v1.5.
3. **top_k** — 1, 3, 5, 7, 10, 15, 20.

The best-measured combination was shipped: **hierarchical chunking, `bge-small-en-v1.5`,
`top_k=10`**.

| | before | now |
|---|---|---|
| retrieval hit@k | 0.38 | **0.81** |
| answers correct | 9/20 | **15/20** |
| wrongly refused to answer | 50% | **19%** |
| hallucinations / bad citations | 0 | 0 |
| cost per 1,000 questions | $0.62 | $1.92 |

McNemar's exact test on the same questions: 7–0 on retrieval (p = 0.016), 6–0 on
correctness (p = 0.031).

**Guardrails.** 12 prompt-injection attacks, half planted inside retrieved documents:
all held, no system-prompt leak, no invented articles ([`eval/red_team.py`](backend/eval/red_team.py)).

**CI.** Every pull request runs a regression gate — parse, chunk and retrieval, no LLM
calls — that fails if any metric drops below the committed baseline.

**Caveats.** 20 questions is a small sample (±20pp on absolute numbers), which is why
comparisons are paired. Answer-correctness numbers come from an LLM judge that has not yet
been validated against human labels.

Full report: [`backend/eval/results/REPORT.md`](backend/eval/results/REPORT.md).

```bash
cd backend
pip install -r requirements-dev.txt
python -m eval.gate                          # regression gate, no LLM
python -m eval.ablations.embeddings --fast   # encoder sweep, no LLM
python -m eval.ablations.chunking --full     # + generation and judge (needs GEMINI_API_KEY)
python -m eval.red_team                      # prompt-injection suite
python -m eval.report
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
