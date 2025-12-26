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
        # Generate embeddings
        embeddings = self.model.encode(texts, show_progress_bar=True)
        self.embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Use IndexFlatIP for inner product (cosine similarity with normalized vectors)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(self.embeddings.astype('float32'))
        
        print(f"Added {len(chunks)} chunks to vector store")
    
    def search(self, query: str, top_k: int = 3) -> List[Tuple[Dict[str, str], float]]:
        """
        Search for relevant chunks by query
        
        Args:
            query: search query
            top_k: number of results
            
        Returns:
            List of tuples (chunk, similarity_score) where score is 0-100%
        """
        if self.index is None:
            raise ValueError("Index not initialized. Add chunks using add_chunks()")
        
        # Generate and normalize query embedding
        query_embedding = self.model.encode([query])
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        # Search returns cosine similarity scores (range: -1 to 1, typically 0.5 to 1.0 for relevant texts)
        scores, indices = self.index.search(
            query_embedding.astype('float32'), 
            top_k
        )

        print("scores", scores)
        
        results = []
        for idx, score in zip(indices[0], scores[0]):
            # Convert cosine similarity (-1 to 1) to percentage (0-100%)
            # For text similarity, we typically see values between 0 and 1
            # So we just multiply by 100
            similarity_percentage = max(0.0, float(score) * 100)
            results.append((self.chunks[idx], similarity_percentage))
        
        return results