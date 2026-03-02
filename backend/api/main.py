from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.rag_system import RAGDemo
from config import GEMINI_API_KEY

app = FastAPI(title="Legal RAG API")

frontend_url_regex = r"https://legal-rag-demo.*\.vercel\.app|http://localhost:3000"
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=frontend_url_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_system = RAGDemo(gemini_api_key=GEMINI_API_KEY)

@app.on_event("startup")
async def startup_event():
    try:
        print("\n" + "=" * 70)
        print("STARTUP: Preloading GDPR embeddings...")
        print("=" * 70)

        gdpr_path = "example_data/gdpr.pdf"
        embeddings_path = "example_data/gdpr_faiss_index"  # folder, not .pkl

        if os.path.exists(gdpr_path):
            if os.path.exists(embeddings_path):
                rag_system.load_index(embeddings_path)
                print("✓ GDPR embeddings loaded from cache")
            else:
                rag_system.setup(pdf_path=gdpr_path)
                rag_system.save_index(embeddings_path)
                print("✓ GDPR embeddings created and cached")
        else:
            print(f"⚠ Warning: GDPR file not found at {gdpr_path}")

        print("=" * 70 + "\n")
    except Exception as e:
        print(f"⚠ Warning: Could not preload GDPR: {str(e)}\n")


class QuestionRequest(BaseModel):
    question: str
    top_k: int = 3


@app.get("/")
def root():
    return {"message": "Legal RAG API", "status": "running"}


@app.post("/api/load-gdpr")
async def load_gdpr():
    try:
        embeddings_path = "example_data/gdpr_faiss_index"
        if os.path.exists(embeddings_path):
            rag_system.load_index(embeddings_path)
        else:
            rag_system.setup(pdf_path="example_data/gdpr.pdf")
            rag_system.save_index(embeddings_path)

        return {"success": True, "document": "gdpr.pdf"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload-document")
async def upload_document(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    try:
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)

        rag_system.setup(pdf_path=temp_path)
        os.remove(temp_path)

        return {"success": True, "document": file.filename}
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ask")
async def ask_question(request: QuestionRequest):
    if rag_system.retriever is None:
        raise HTTPException(status_code=400, detail="No document loaded")

    try:
        result = rag_system.answer(request.question)
        return {"answer": result["answer"], "chunks": result["chunks"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "rag_initialized": rag_system is not None,
        "document_loaded": rag_system.retriever is not None,
    }