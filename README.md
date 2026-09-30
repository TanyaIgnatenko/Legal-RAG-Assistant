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

Eval harness in [`backend/eval/`](backend/eval). 20 GDPR questions across all 11 chapters,
each labelled with the exact sentence that answers it. Retrieval is scored by character
offsets, so different chunkers and encoders are comparable.

**The production config was chosen by measurement**, not by default: hierarchical chunking,
`bge-small-en-v1.5`, `top_k=10`.

| | before | now |
|---|---|---|
| retrieval hit@k | 0.38 | **0.81** |
| answers correct | 9/20 | **15/20** |
| wrongly refused to answer | 50% | **19%** |
| hallucinations / bad citations | 0 | 0 |
| cost per 1,000 questions | $0.62 | $1.92 |

McNemar's exact test on the same questions: 7–0 on retrieval (p = 0.016), 6–0 on
correctness (p = 0.031).

**How the bottleneck was found.** With the correct passage handed to it, the model answers
18/20; with no context, 2/20. So generation was fine and every loss was a retrieval miss.
Traces record the rank of the correct passage, which separated "top_k too small" from "the
encoder never finds it" — the old multilingual encoder missed 2 of 16 questions at any
depth, the new one misses none.

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
