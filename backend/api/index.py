from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os

# Add parent directory to path to import RAG system
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.rag_system import RAGDemo
from config import GEMINI_API_KEY, CORS_ORIGINS

app = FastAPI(title="Legal RAG API")

# CORS middleware - allow frontend to communicate
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,  # Vite default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG instance
rag_system = RAGDemo(gemini_api_key=GEMINI_API_KEY)

class QuestionRequest(BaseModel):
    question: str
    top_k: int = 3

class DocumentInfo(BaseModel):
    name: str
    chunks: int

@app.get("/")
def root():
    return {"message": "Legal RAG API", "status": "running"}

@app.post("/api/load-gdpr")
async def load_gdpr():
    """Load pre-loaded GDPR document"""
    global rag_system
    
    try:
        print("1")
        rag_system.setup("example_data/gdpr.pdf")
        print("2")
        chunks = len(rag_system.vector_store.chunks)
        print("3")
        
        return {
            "success": True,
            "document": "gdpr.pdf",
            "chunks": chunks
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload-document")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process custom document"""
    global rag_system
    
    try:
        # Save uploaded file temporarily
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Process with RAG system
        rag_system.setup(temp_path)
        chunks = len(rag_system.vector_store.chunks)
        
        # Clean up
        os.remove(temp_path)
        
        return {
            "success": True,
            "document": file.filename,
            "chunks": chunks
        }
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ask")
async def ask_question(request: QuestionRequest):
    """Ask question about loaded document"""
    global rag_system
    
    if not rag_system or not rag_system.vector_store.chunks:
        raise HTTPException(status_code=400, detail="No document loaded")
    
    try:
        # Get answer
        answer = rag_system.answer(
            request.question,
            top_k=request.top_k,
            verbose=False
        )
        
        # Get chunks
        chunks = rag_system.vector_store.search(
            request.question,
            top_k=request.top_k
        )
        
        # Format chunks for response
        formatted_chunks = [
            {
                "metadata": chunk["metadata"],
                "text": chunk["text"],
                "distance": float(distance)
            }
            for chunk, distance in chunks
        ]
        
        return {
            "answer": answer,
            "chunks": formatted_chunks
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "rag_initialized": rag_system is not None,
        "document_loaded": rag_system.vector_store.chunks if rag_system else []
    }