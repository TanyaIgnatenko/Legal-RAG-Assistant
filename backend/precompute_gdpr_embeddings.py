"""
Precompute the FAISS index for the GDPR document.

Run this once to generate the index that the API loads on startup, instead of
re-embedding 99 chunks every time the server boots. Re-run it whenever the
chunker or the embedding model changes — chunk metadata and the start/end
offsets are baked into the saved index.

Usage:
    python precompute_gdpr_embeddings.py
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.rag_system import DEFAULT_EMBEDDING_MODEL, RAGDemo  # noqa: E402


def precompute_index(
    pdf_path: str = "example_data/gdpr.pdf",
    output_path: str = "example_data/gdpr_faiss_index",
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> bool:
    """Parse, chunk, embed and save the GDPR index."""
    print("=" * 70)
    print("PRECOMPUTING GDPR INDEX")
    print("=" * 70)

    if not os.path.exists(pdf_path):
        print(f"❌ Error: PDF file not found at {pdf_path}")
        return False

    try:
        # No API key needed: this path never calls the LLM.
        rag = RAGDemo(gemini_api_key="", embedding_model=model_name)
        rag.setup(pdf_path=pdf_path)
        rag.save_index(output_path)

        # Reload into the same instance: a second RAGDemo would load a second
        # copy of the encoder, which does not fit alongside the first.
        print(f"\nVerifying {output_path} ...")
        rag.vectorstore = None
        rag.load_index(output_path)
        stored = rag.vectorstore.docstore._dict
        sample = next(iter(stored.values()))

        print(f"   ✓ {len(stored)} chunks indexed")
        print(f"   ✓ model: {model_name}")
        print(f"   ✓ sample source: {sample.metadata.get('source')!r}")
        print(f"   ✓ sample span:   "
              f"({sample.metadata.get('start')}, {sample.metadata.get('end')})")

        if sample.metadata.get("start") is None:
            print("   ⚠ Warning: chunks carry no offsets — eval metrics need them.")

        print("\n" + "=" * 70)
        print("✅ PRECOMPUTATION COMPLETE")
        print("=" * 70)
        return True

    except Exception as e:
        print(f"\n❌ Error during precomputation: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    sys.exit(0 if precompute_index() else 1)
