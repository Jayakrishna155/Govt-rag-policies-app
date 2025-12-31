"""
Text Splitter Module
Handles text chunking - lightweight implementation without heavy dependencies
"""
from typing import List


class SimpleTextSplitter:
    """
    Simple recursive text splitter that doesn't require transformers/torch.
    Mimics RecursiveCharacterTextSplitter behavior.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, separators: List[str] = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks recursively."""
        if not text or len(text.strip()) == 0:
            return []
        
        if len(text) <= self.chunk_size:
            return [text.strip()] if text.strip() else []
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Find the end of the chunk
            end = start + self.chunk_size
            
            if end >= len(text):
                # Last chunk
                chunk = text[start:].strip()
                if chunk:
                    chunks.append(chunk)
                break
            
            # Try to find a good split point using separators
            best_split = end
            found_split = False
            
            for separator in self.separators:
                if separator == "":
                    # Character-level split (last resort)
                    best_split = end
                    found_split = True
                    break
                
                # Look for separator near the end
                search_start = max(start, end - self.chunk_overlap)
                search_end = min(len(text), end + self.chunk_overlap)
                search_text = text[search_start:search_end]
                
                # Find last occurrence of separator
                if separator in search_text:
                    last_pos = search_text.rfind(separator)
                    if last_pos != -1:
                        best_split = search_start + last_pos + len(separator)
                        found_split = True
                        break
            
            # Extract chunk
            chunk = text[start:best_split].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            if found_split:
                start = max(start + 1, best_split - self.chunk_overlap)
            else:
                start = best_split
        
        return chunks


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Split text into chunks using simple recursive splitter.
    
    Args:
        text: Input text to chunk
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of text chunks
    """
    splitter = SimpleTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = splitter.split_text(text)
    return chunks


def chunk_text_with_metadata(
    text: str, 
    page_num: int, 
    filename: str,
    chunk_size: int = 1000, 
    chunk_overlap: int = 200
) -> List[dict]:
    """
    Split text into chunks with metadata.
    
    Args:
        text: Input text to chunk
        page_num: Page number
        filename: Source filename
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of dictionaries with 'text', 'page', and 'filename' keys
    """
    chunks = chunk_text(text, chunk_size, chunk_overlap)
    
    chunk_metadata = []
    for chunk in chunks:
        chunk_metadata.append({
            'text': chunk,
            'page': page_num,
            'filename': filename
        })
    
    return chunk_metadata
