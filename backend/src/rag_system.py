"""RAG system implementation for legal document Q&A"""

import google.generativeai as genai
from typing import List
from .parser import PDFParser
from .chunker import LegalChunker
from .vector_store import FaissVectorStore


class RAGDemo:
    """RAG system demo for legal documents"""
    
    def __init__(self, gemini_api_key: str):
        """
        Initialize RAG system
        
        Args:
            gemini_api_key: API key for Gemini
        """
        self.vector_store = FaissVectorStore()
        self.parser = PDFParser()
        self.chunker = LegalChunker()
        
        try:
            genai.configure(api_key=gemini_api_key)
            self.model = genai.GenerativeModel('gemini-2.5-flash')
            print("✓ Gemini model initialized successfully")
        except Exception as e:
            print(f"⚠ Warning: Failed to initialize Gemini model: {str(e)}")
            self.model = None
    
    def setup(self, pdf_path: str):
        """
        System setup: parsing, chunking, indexing
        
        Args:
            pdf_path: path to PDF file
        """
        print("=" * 60)
        print("RAG SYSTEM SETUP")
        print("=" * 60)
        
        print("\n1. Parsing PDF document...")
        text = self.parser.parse(pdf_path)
        print(f"   Extracted {len(text)} characters")
        
        print("\n2. Hierarchical chunking...")
        chunks = self.chunker.chunk_gdpr(text)
        print(f"   Created {len(chunks)} chunks")
        
        print("\n3. Creating vector index...")
        self.vector_store.add_chunks(chunks)
        
        print("\n" + "=" * 60)
        print("SYSTEM READY")
        print("=" * 60 + "\n")
    
    def answer(self, question: str, top_k: int = 3, verbose: bool = True) -> str:
        """
        Answer question using RAG
        
        Args:
            question: user question
            top_k: number of relevant chunks
            verbose: whether to print details
            
        Returns:
            System answer
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"QUESTION: {question}")
            print(f"{'='*60}\n")
        
        results = self.vector_store.search(question, top_k=top_k)
        
        if verbose:
            print("Found relevant fragments:\n")
            for i, (chunk, dist) in enumerate(results, 1):
                print(f"{i}. {chunk['metadata']} (distance: {dist:.4f})")
                print(f"   {chunk['text'][:200]}...\n")
        
        context = "\n\n".join([
            f"[{chunk['metadata']}]\n{chunk['text']}" 
            for chunk, _ in results
        ])
        
        prompt = f"""You are a legal document expert. Answer the question based on the provided context.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Answer only based on the provided context
- If information is insufficient, say so
- Reference specific articles and chapters
- Be precise and specific

ANSWER:"""
        
        try:
            if self.model is None:
                answer = "Error: Gemini model not initialized. Check your API key."
            else:
                response = self.model.generate_content(prompt)
                answer = response.text
        except Exception as e:
            answer = f"Error generating answer: {str(e)}"
        
        if verbose:
            print(f"{'='*60}")
            print("ANSWER:")
            print(f"{'='*60}")
            print(answer)
            print(f"{'='*60}\n")
        
        return answer