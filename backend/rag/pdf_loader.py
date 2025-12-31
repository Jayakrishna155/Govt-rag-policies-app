"""
PDF Loader Module
Handles PDF text extraction and processing
"""
import hashlib
from typing import List, Tuple
from pypdf import PdfReader
import io


def extract_text_from_pdf(pdf_bytes: bytes) -> Tuple[str, str]:
    """
    Extract text from PDF bytes and generate document hash.
    
    Args:
        pdf_bytes: PDF file as bytes
        
    Returns:
        Tuple of (extracted_text, document_hash)
    """
    try:
        # Create PDF reader from bytes
        pdf_file = io.BytesIO(pdf_bytes)
        reader = PdfReader(pdf_file)
        
        # Extract text from all pages
        text_parts = []
        for page in reader.pages:
            text = page.extract_text()
            if text.strip():
                text_parts.append(text)
        
        full_text = "\n\n".join(text_parts)
        
        # Generate SHA256 hash of the PDF content
        document_hash = hashlib.sha256(pdf_bytes).hexdigest()
        
        return full_text, document_hash
    
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {str(e)}")


def extract_text_with_pages(pdf_bytes: bytes) -> List[Tuple[str, int]]:
    """
    Extract text from PDF with page numbers.
    
    Args:
        pdf_bytes: PDF file as bytes
        
    Returns:
        List of tuples (text, page_number)
    """
    try:
        pdf_file = io.BytesIO(pdf_bytes)
        reader = PdfReader(pdf_file)
        
        page_texts = []
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text.strip():
                page_texts.append((text, page_num))
        
        return page_texts
    
    except Exception as e:
        raise Exception(f"Error extracting text with pages: {str(e)}")

