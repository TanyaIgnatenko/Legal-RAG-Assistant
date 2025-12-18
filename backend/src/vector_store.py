"""Vector store module using FAISS for similarity search"""

import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import faiss


class FaissVectorStore:
    """Vector store based on FAISS"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize vector store
        
        Args:
            model_name: name of the model for embeddings
        """
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.chunks = []
        self.embeddings = []
    
    def add_chunks(self, chunks: List[Dict[str, str]]):
        """
        Add chunks to vector store
        
        Args:
            chunks: list of chunks with metadata
        """
        self.chunks = chunks
        texts = [chunk['text'] for chunk in chunks]
        
        print(f"Generating embeddings for {len(texts)} chunks...")
        self.embeddings = self.model.encode(texts, show_progress_bar=True)
        
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(np.array(self.embeddings).astype('float32'))
        
        print(f"Added {len(chunks)} chunks to vector store")
    
    def search(self, query: str, top_k: int = 3) -> List[Tuple[Dict[str, str], float]]:
        """
        Search for relevant chunks by query
        
        Args:
            query: search query
            top_k: number of results
            
        Returns:
            List of tuples (chunk, distance)
        """
        if self.index is None:
            raise ValueError("Index not initialized. Add chunks using add_chunks()")
        
        query_embedding = self.model.encode([query])
        
        distances, indices = self.index.search(
            np.array(query_embedding).astype('float32'), 
            top_k
        )
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            results.append((self.chunks[idx], float(dist)))
        
        return results