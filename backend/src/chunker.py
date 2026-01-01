"""Text chunking module for legal documents"""

import re
from typing import List, Dict


class LegalChunker:
    """Hierarchical chunking for legal documents"""
    
    @staticmethod
    def chunk_gdpr(text: str) -> List[Dict[str, str]]:
        """
        Splits GDPR text into chunks by chapters and articles
        
        Args:
            text: document text
            min_chunk_size: minimum character length for chunks
            
        Returns:
            List of chunks with metadata
        """
        chunks = []
        
        # Patterns for finding chapters and articles
        chapter_pattern = r'(?:CHAPTER)\s+(?:[IVXLCDM]+|\d+)(?:\s*[-:.]?\s*[^\n]*)?'
        # Article pattern - MUST be at start of line to avoid false matches in text
        article_pattern = r'(?:^|\n)\s*(?:Article)\s+\d+\s\n'
        
        # Find all chapters
        chapter_matches = list(re.finditer(chapter_pattern, text))
        
        for i, chapter_match in enumerate(chapter_matches):
            chapter_start = chapter_match.start()
            chapter_end = chapter_matches[i + 1].start() if i + 1 < len(chapter_matches) else len(text)
            chapter_text = text[chapter_start:chapter_end]
            chapter_name = chapter_match.group(0)
            
            # Find articles in chapter
            article_matches = list(re.finditer(article_pattern, chapter_text, re.IGNORECASE))
            
            if article_matches:
                for j, article_match in enumerate(article_matches):
                    article_start = article_match.start()
                    article_end = article_matches[j + 1].start() if j + 1 < len(article_matches) else len(chapter_text)
                    article_text = chapter_text[article_start:article_end].strip()
                    article_name = article_match.group(0)
                    
                    chunks.append({
                        'chapter': chapter_name,
                        'article': article_name,
                        'text': article_text,
                        'metadata': f"{chapter_name} - {article_name}"
                    })
            else:
                chapter_text.strip()
                chunks.append({
                    'chapter': chapter_name,
                    'article': 'N/A',
                    'text': chapter_text.strip(),
                    'metadata': chapter_name
                })
        
        # If no chapters found, split text into paragraphs
        if not chunks:
            paragraphs = text.split('\n\n')
            for i, para in enumerate(paragraphs):
                para.strip()
                chunks.append({
                    'chapter': 'N/A',
                    'article': f'Section {i+1}',
                    'text': para.strip(),
                    'metadata': f'Section {i+1}'
                })
        
        return chunks