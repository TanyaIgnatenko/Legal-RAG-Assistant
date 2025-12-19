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
Generate Embeddings (all-MiniLM-L6-v2)
    ↓
Store in Vector DB (FAISS)
    ↓
User Query → Semantic Search → Retrieve Top-K Chunks
    ↓
LLM (Gemini) + Context → Generate Answer
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
[GitHub](https://github.com/TanyaIgnatenko) • [LinkedIn](www.linkedin.com/in/tatyana-ignatenko)
