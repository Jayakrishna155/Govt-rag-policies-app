"""
Duplicate Handler Module
Manages document duplicate detection using SHA256 hashing
"""
import json
import os
from typing import Set
from pathlib import Path


class DuplicateHandler:
    """Handles duplicate document detection and tracking"""
    
    def __init__(self, metadata_file: str = "data/uploaded_docs.json"):
        """
        Initialize duplicate handler.
        
        Args:
            metadata_file: Path to JSON file storing document metadata
        """
        self.metadata_file = metadata_file
        self.uploaded_hashes: Set[str] = set()
        self._load_metadata()
    
    def _load_metadata(self):
        """Load existing document metadata from file"""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.uploaded_hashes = set(data.get('hashes', []))
            except Exception as e:
                print(f"Warning: Could not load metadata: {e}")
                self.uploaded_hashes = set()
        else:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.metadata_file), exist_ok=True)
            self.uploaded_hashes = set()
    
    def is_duplicate(self, document_hash: str) -> bool:
        """
        Check if document hash already exists.
        
        Args:
            document_hash: SHA256 hash of the document
            
        Returns:
            True if duplicate, False otherwise
        """
        return document_hash in self.uploaded_hashes
    
    def add_document(self, document_hash: str, filename: str):
        """
        Add document hash to tracked documents.
        
        Args:
            document_hash: SHA256 hash of the document
            filename: Name of the uploaded file
        """
        if document_hash not in self.uploaded_hashes:
            self.uploaded_hashes.add(document_hash)
            self._save_metadata()
    
    def _save_metadata(self):
        """Save document metadata to file"""
        try:
            data = {
                'hashes': list(self.uploaded_hashes),
                'count': len(self.uploaded_hashes)
            }
            os.makedirs(os.path.dirname(self.metadata_file), exist_ok=True)
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save metadata: {e}")

