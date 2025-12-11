A Retrieval-Augmented Generation (RAG) system for analyzing legal documents using AI.

## Features

- 📄 PDF document parsing
- 🔍 Hierarchical chunking for legal documents
- 🧠 Semantic search using FAISS
- 🤖 AI-powered Q&A with Google Gemini
- 💬 Interactive chat interface

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/legal-rag-demo.git
cd legal-rag-demo
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your API key:
   - Option 1: Edit `config.py` and add your Gemini API key
   - Option 2: Set environment variable: `export GEMINI_API_KEY="your-key"`

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Project Structure

```
legal-rag-demo/
├── src/
│   ├── __init__.py
│   ├── parser.py          # PDF parsing
│   ├── chunker.py         # Text chunking
│   ├── vector_store.py    # Vector search
│   └── rag_system.py      # RAG implementation
├── app.py                 # Streamlit frontend
├── config.py              # Configuration
└── requirements.txt       # Dependencies
```

## License

MIT License

## Author

Tatyana Ignatenko