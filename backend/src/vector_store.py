"""Vector store module using FAISS for similarity search"""

import numpy as np
import pickle
import os
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
import faiss


class FaissVectorStore:
    """Vector store based on FAISS"""
    
    def __init__(self, model_name: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
        """
        Initialize vector store
        
        Args:
            model_name: name of the model for embeddings
        """
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.model_name = model_name
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
    
    def load_precomputed(self, embeddings_path: str) -> bool:
        """
        Load precomputed embeddings from file
        
        Args:
            embeddings_path: path to pickled embeddings file
            
        Returns:
            True if loaded successfully, False otherwise
        """
        if not os.path.exists(embeddings_path):
            print(f"⚠ Precomputed embeddings not found at: {embeddings_path}")
            return False
        
        try:
            print(f"Loading precomputed embeddings from: {embeddings_path}")
            
            with open(embeddings_path, 'rb') as f:
                data = pickle.load(f)
            
            # Verify model compatibility
            if data['model_name'] != self.model_name:
                print(f"⚠ Warning: Embeddings were created with {data['model_name']}, "
                      f"but current model is {self.model_name}")
                return False
            
            # Load data
            self.chunks = data['chunks']
            self.embeddings = data['embeddings']
            self.dimension = data['dimension']
            
            # Rebuild FAISS index
            self.index = faiss.IndexFlatIP(self.dimension)
            self.index.add(self.embeddings.astype('float32'))
            
            print(f"✓ Loaded {len(self.chunks)} precomputed chunks")
            return True
            
        except Exception as e:
            print(f"❌ Error loading precomputed embeddings: {str(e)}")
            return False
    
    def save_embeddings(self, output_path: str):
        """
        Save current embeddings to file
        
        Args:
            output_path: path to save embeddings
        """
        if not self.chunks or len(self.embeddings) == 0:
            raise ValueError("No embeddings to save. Call add_chunks() first.")
        
        # Create directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        embedding_data = {
            'chunks': self.chunks,
            'embeddings': self.embeddings,
            'model_name': self.model_name,
            'dimension': self.dimension,
            'num_chunks': len(self.chunks)
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(embedding_data, f)
        
        print(f"✓ Saved embeddings to: {output_path}")
    
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
            raise ValueError("Index not initialized. Add chunks using add_chunks() or load_precomputed()")
        
        # Generate and normalize query embedding
        query_embedding = self.model.encode([query])
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        # Search returns cosine similarity scores (range: -1 to 1, typically 0.5 to 1.0 for relevant texts)
        scores, indices = self.index.search(
            query_embedding.astype('float32'), 
            top_k
        )
        
        results = []
        for idx, score in zip(indices[0], scores[0]):
            # Skip invalid indices (FAISS uses -1 for padding when fewer results than top_k)
            if idx == -1:
                continue

            # Convert cosine similarity (-1 to 1) to percentage (0-100%)
            # For text similarity, we typically see values between 0 and 1
            # So we just multiply by 100
            similarity_percentage = max(0.0, float(score) * 100)
            results.append((self.chunks[idx], similarity_percentage))

        return results