# Legal RAG Demo

Legal RAG Assistant to help you analyze legal documents more easily and accurately.

[![Live Demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)

## 📌 Goal

Demonstrate understanding of RAG architecture in Legal Tech context with a working prototype.

## 🚀 Quick Start

```bash
# 1. Clone repository
git clone https://github.com/yourusername/legal-rag-demo.git
cd legal-rag-demo

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set API key in config.toml
GEMINI_API_KEY="your-key-here"

# 4. Run
streamlit run app.py
```

Open http://localhost:8501

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

- **Backend**: Python, PyMuPDF, Sentence Transformers, FAISS
- **LLM**: Google Gemini API
- **Frontend**: Streamlit

## 📁 Structure

```
legal-rag-demo/
├── src/
│   ├── parser.py          # PDF parsing
│   ├── chunker.py         # Text chunking
│   ├── vector_store.py    # FAISS search
│   └── rag_system.py      # RAG pipeline
├── app.py                 # Streamlit UI
├── config.py              # Configuration
└── requirements.txt       # Dependencies
```

## 💡 Usage

1. Upload a PDF document (e.g., GDPR) or use pre-loaded one
2. Ask questions in natural language
3. Get AI answers with source citations

**Example questions:**
- What is personal data according to GDPR?
- What are the penalties for violations?
- What are data subject rights?

## 🎥 Demo

**Live:** [Try it here](https://your-app.streamlit.app)

## 📝 License

MIT

## 👤 Author

**Your Name**  
[GitHub](https://github.com/TanyaIgnatenko) • [LinkedIn](www.linkedin.com/in/tatyana-ignatenko)