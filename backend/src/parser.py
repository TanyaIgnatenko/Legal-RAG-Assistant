import pymupdf

class PDFParser:
    """Parser for extracting text from PDF documents"""
    
    @staticmethod
    def parse(pdf_path: str) -> str:
        """Extract text from PDF file"""
        doc = pymupdf.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text