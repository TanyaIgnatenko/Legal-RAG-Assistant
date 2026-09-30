"""Legal RAG Demo package"""

from .parser import PDFParser
from .chunker import (
    Chunker,
    HierarchicalChunker,
    LegalChunker,
    RecursiveOverlapChunker,
    WholeDocumentChunker,
    get_chunker,
)
from .rag_system import RAGDemo

__all__ = [
    'PDFParser',
    'Chunker',
    'HierarchicalChunker',
    'RecursiveOverlapChunker',
    'WholeDocumentChunker',
    'LegalChunker',
    'get_chunker',
    'RAGDemo',
]
__version__ = '1.0.0'
