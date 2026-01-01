"""
Precompute embeddings for GDPR document

Run this script once to generate embeddings that will be loaded on app startup.
This avoids recomputing embeddings every time the server starts.

Usage:
    python precompute_gdpr_embeddings.py
"""

import sys
import os
import pickle
from pathlib import Path

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.parser import PDFParser
from src.chunker import LegalChunker
from src.vector_store import FaissVectorStore


def precompute_embeddings(
    pdf_path: str = "example_data/gdpr.pdf",
    output_path: str = "example_data/gdpr_embeddings.pkl",
    model_name: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
):
    """
    Precompute and save embeddings for GDPR document
    
    Args:
        pdf_path: Path to GDPR PDF file
        output_path: Path to save embeddings
        model_name: Embedding model to use
    """
    print("=" * 70)
    print("PRECOMPUTING GDPR EMBEDDINGS")
    print("=" * 70)
    
    # Check if PDF exists
    if not os.path.exists(pdf_path):
        print(f"❌ Error: PDF file not found at {pdf_path}")
        return False
    
    try:
        # Step 1: Parse PDF
        print(f"\n1. Parsing PDF: {pdf_path}")
        parser = PDFParser()
        text = parser.parse(pdf_path)
        print(f"   ✓ Extracted {len(text):,} characters")
        
        # Step 2: Chunk document
        print("\n2. Chunking document...")
        chunker = LegalChunker()
        chunks = chunker.chunk_gdpr(text)
        print(f"   ✓ Created {len(chunks)} chunks")
        
        # Step 3: Generate embeddings
        print(f"\n3. Generating embeddings using model: {model_name}")
        vector_store = FaissVectorStore(model_name=model_name)
        vector_store.add_chunks(chunks)
        print(f"   ✓ Generated embeddings (dimension: {vector_store.dimension})")
        
        # Step 4: Save to file
        print(f"\n4. Saving embeddings to: {output_path}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save all necessary data
        embedding_data = {
            'chunks': vector_store.chunks,
            'embeddings': vector_store.embeddings,
            'model_name': model_name,
            'dimension': vector_store.dimension,
            'pdf_path': pdf_path,
            'num_chunks': len(chunks)
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(embedding_data, f)
        
        # Get file size
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(f"   ✓ Saved embeddings ({file_size:.2f} MB)")
        
        # Verify by loading
        print("\n5. Verifying saved embeddings...")
        with open(output_path, 'rb') as f:
            loaded_data = pickle.load(f)
        print(f"   ✓ Verification successful")
        print(f"   - Chunks: {loaded_data['num_chunks']}")
        print(f"   - Model: {loaded_data['model_name']}")
        print(f"   - Dimension: {loaded_data['dimension']}")
        
        print("\n" + "=" * 70)
        print("✅ PRECOMPUTATION COMPLETE")
        print("=" * 70)
        print(f"\nEmbeddings saved to: {output_path}")
        print("The FastAPI app will now load these precomputed embeddings on startup.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during precomputation: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = precompute_embeddings()
    sys.exit(0 if success else 1)
