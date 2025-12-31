"""
Vector Store Module
Manages FAISS vector database with persistence and metadata
"""
import os
import json
import numpy as np
import faiss
from typing import List, Dict, Tuple
from pathlib import Path


class FAISSVectorStore:
    """Manages FAISS vector database with metadata tracking"""
    
    def __init__(self, index_path: str = "data/faiss_index", metadata_path: str = "data/faiss_metadata.json"):
        """
        Initialize FAISS vector store.
        
        Args:
            index_path: Directory path for FAISS index files
            metadata_path: Path to JSON file storing metadata
        """
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index = None
        self.embedding_dim = 384  # Dimension for all-MiniLM-L6-v2
        self.metadata: List[Dict] = []
        self._load_or_create_index()
    
    def _load_or_create_index(self):
        """Load existing FAISS index or create new one"""
        os.makedirs(self.index_path, exist_ok=True)
        
        index_file = os.path.join(self.index_path, "index.faiss")
        if os.path.exists(index_file):
            try:
                # Load existing index
                self.index = faiss.read_index(index_file)
                # Update embedding_dim to match the loaded index
                self.embedding_dim = self.index.d
                self._load_metadata()
                print(f"Loaded existing FAISS index with {self.index.ntotal} vectors (dimension: {self.embedding_dim})")
            except Exception as e:
                print(f"Error loading index: {e}. Creating new index.")
                self._create_new_index()
        else:
            self._create_new_index()
    
    def _create_new_index(self):
        """Create a new FAISS index"""
        # Using L2 distance (Euclidean) - can also use InnerProduct for cosine similarity
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.metadata = []
        print("Created new FAISS index")
    
    def _load_metadata(self):
        """Load metadata from JSON file"""
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load metadata: {e}")
                self.metadata = []
        else:
            self.metadata = []
    
    def _save_metadata(self):
        """Save metadata to JSON file"""
        try:
            os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Could not save metadata: {e}")
    
    def add_vectors(self, embeddings: np.ndarray, chunk_metadata: List[Dict]):
        """
        Add vectors and metadata to the index.
        
        Args:
            embeddings: Numpy array of embeddings (shape: [num_vectors, embedding_dim])
            chunk_metadata: List of metadata dicts for each chunk
        """
        if self.index is None:
            self._create_new_index()
        
        # Ensure embeddings are float32
        embeddings = embeddings.astype('float32')
        
        # Check dimension compatibility
        if len(embeddings.shape) != 2:
            raise ValueError(f"Embeddings must be 2D array, got shape {embeddings.shape}")
        
        actual_dim = embeddings.shape[1]
        if actual_dim != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.embedding_dim}, got {actual_dim}. "
                f"This usually happens when the embedding model dimension changes. "
                f"Please delete the existing FAISS index and restart."
            )
        
        # Add vectors to index
        self.index.add(embeddings)
        
        # Add metadata
        for meta in chunk_metadata:
            self.metadata.append(meta)
        
        # Save index and metadata
        self.save()
    
    def search(self, query_embedding: np.ndarray, k: int = 4) -> List[Tuple[Dict, float]]:
        """
        Search for similar vectors.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of tuples (metadata_dict, distance_score)
        """
        if self.index is None or self.index.ntotal == 0:
            return []
        
        # Ensure query embedding is float32 and reshape
        query_embedding = query_embedding.astype('float32').reshape(1, -1)
        
        # Search
        distances, indices = self.index.search(query_embedding, min(k, self.index.ntotal))
        
        # Retrieve metadata for results
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.metadata):
                results.append((self.metadata[idx], float(dist)))
        
        return results
    
    def save(self):
        """Save FAISS index to disk"""
        if self.index is not None:
            try:
                index_file = os.path.join(self.index_path, "index.faiss")
                faiss.write_index(self.index, index_file)
                self._save_metadata()
                print(f"Saved FAISS index with {self.index.ntotal} vectors")
            except Exception as e:
                print(f"Error saving index: {e}")
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector store"""
        return {
            "total_vectors": self.index.ntotal if self.index else 0,
            "embedding_dim": self.embedding_dim,
            "metadata_count": len(self.metadata)
        }

