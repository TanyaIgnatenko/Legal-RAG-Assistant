"""Legal RAG Demo package"""

from .parser import PDFParser
from .chunker import LegalChunker
from .vector_store import FaissVectorStore
from .rag_system import RAGDemo

__all__ = ['PDFParser', 'LegalChunker', 'FaissVectorStore', 'RAGDemo']
__version__ = '1.0.0'v