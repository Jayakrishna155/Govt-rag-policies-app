"""
Embeddings Module
Handles text embeddings using TF-IDF + SVD (torch-free implementation)
"""
import numpy as np
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import pickle
import os


class EmbeddingModel:
    """Manages TF-IDF + SVD embedding model (no torch/transformers dependency)"""
    
    def __init__(self, embedding_dim: int = 384, model_path: str = "data/embedding_model.pkl"):
        """
        Initialize embedding model.
        
        Args:
            embedding_dim: Dimension of output embeddings (default 384 to match all-MiniLM-L6-v2)
            model_path: Path to save/load the fitted model
        """
        self.embedding_dim = embedding_dim
        self.model_path = model_path
        self.vectorizer = None
        self.svd = None
        self._initialize_model()
        self._load_model_if_exists()
    
    def _initialize_model(self):
        """Initialize TF-IDF vectorizer and SVD model"""
        try:
            print("Initializing TF-IDF + SVD embedding model...")
            # TF-IDF vectorizer with reasonable max features
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),  # unigrams and bigrams
                stop_words='english',
                lowercase=True,
                min_df=1,
                max_df=1.0  # Changed to 1.0 to avoid issues with small document sets
            )
            
            # SVD for dimensionality reduction
            self.svd = TruncatedSVD(n_components=self.embedding_dim, random_state=42)
            print(f"Embedding model initialized (dimension: {self.embedding_dim})")
        except Exception as e:
            raise Exception(f"Error initializing embedding model: {str(e)}")
    
    def _save_model(self):
        """Save the fitted model to disk"""
        if not self.is_fitted():
            return
        
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            model_data = {
                'vectorizer': self.vectorizer,
                'svd': self.svd,
                'embedding_dim': self.embedding_dim
            }
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"Saved embedding model to {self.model_path}")
        except Exception as e:
            print(f"Warning: Could not save embedding model: {e}")
    
    def refit_from_texts(self, texts: List[str]):
        """Refit the model from a list of texts (useful for restoring from FAISS metadata)"""
        if not texts:
            return False
        
        print(f"Refitting embedding model from {len(texts)} text chunks...")
        try:
            self._fit_if_needed(texts)
            if self.is_fitted():
                self._save_model()
                print("Successfully refitted and saved embedding model.")
                return True
            else:
                print("Warning: Refitting completed but model is not fitted.")
                return False
        except Exception as e:
            print(f"Error refitting model: {e}")
            return False
    
    def _load_model_if_exists(self):
        """Load the fitted model from disk if it exists"""
        if not os.path.exists(self.model_path):
            print("No saved embedding model found. Model will be fitted on first PDF upload.")
            return
        
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.vectorizer = model_data.get('vectorizer')
            self.svd = model_data.get('svd')
            if 'embedding_dim' in model_data:
                self.embedding_dim = model_data['embedding_dim']
            
            if self.is_fitted():
                vocab_size = len(self.vectorizer.vocabulary_) if hasattr(self.vectorizer, 'vocabulary_') else 0
                svd_components = self.svd.components_.shape[0] if hasattr(self.svd, 'components_') else 0
                print(f"Loaded fitted embedding model from {self.model_path} (vocab: {vocab_size}, svd_components: {svd_components})")
            else:
                print(f"Warning: Loaded model from {self.model_path} but it appears not to be fitted. Will refit on next upload.")
                self.vectorizer = None
                self.svd = None
                self._initialize_model()
        except Exception as e:
            print(f"Warning: Could not load embedding model from {self.model_path}: {e}")
            print("Model will be fitted on first PDF upload.")
            self.vectorizer = None
            self.svd = None
            self._initialize_model()
    
    def is_fitted(self) -> bool:
        """Check if the model has been fitted - uses same logic as _fit_if_needed"""
        try:
            # Use the same check as _fit_if_needed for consistency
            vectorizer_fitted = hasattr(self.vectorizer, 'vocabulary_') and self.vectorizer.vocabulary_ is not None
            svd_fitted = hasattr(self.svd, 'components_') and self.svd.components_ is not None
            
            # Additional validation
            if vectorizer_fitted:
                try:
                    vocab_size = len(self.vectorizer.vocabulary_)
                    vectorizer_fitted = vocab_size > 0
                except:
                    vectorizer_fitted = False
            
            if svd_fitted:
                try:
                    svd_fitted = len(self.svd.components_.shape) == 2 and self.svd.components_.shape[0] > 0
                except:
                    svd_fitted = False
            
            result = vectorizer_fitted and svd_fitted
            if not result:
                print(f"DEBUG is_fitted: vectorizer={vectorizer_fitted}, svd={svd_fitted}")
                if hasattr(self.vectorizer, 'vocabulary_'):
                    print(f"  vocab exists: {self.vectorizer.vocabulary_ is not None}")
                if hasattr(self.svd, 'components_'):
                    print(f"  components exists: {self.svd.components_ is not None}")
            return result
        except Exception as e:
            print(f"DEBUG is_fitted exception: {e}")
            return False
    
    def _fit_if_needed(self, texts: List[str]):
        """Fit vectorizer and SVD on texts if not already fitted"""
        # Check if vectorizer is fitted
        vectorizer_fitted = hasattr(self.vectorizer, 'vocabulary_') and self.vectorizer.vocabulary_ is not None
        # Check if SVD is fitted
        svd_fitted = hasattr(self.svd, 'components_') and self.svd.components_ is not None
        
        print(f"DEBUG _fit_if_needed: vectorizer_fitted={vectorizer_fitted}, svd_fitted={svd_fitted}, num_texts={len(texts)}")
        
        if not vectorizer_fitted or not svd_fitted:
            if not texts:
                raise ValueError("Cannot fit model on empty text list")
            
            # Ensure we have at least 2 texts for proper fitting (duplicate if only 1)
            if len(texts) == 1:
                texts = texts * 2  # Duplicate to ensure at least 2 samples
            
            # Fit TF-IDF
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            
            # Check dimensions - SVD can't reduce to more components than we have samples/features
            n_samples, n_features = tfidf_matrix.shape
            
            # Ensure we have at least 2 features for SVD
            if n_features < 2:
                raise ValueError(
                    f"TF-IDF produced only {n_features} feature(s), but TruncatedSVD requires at least 2. "
                    f"Please provide more diverse text or upload documents first."
                )
            
            # Use a minimum dimension to ensure FAISS compatibility
            # We'll use min(embedding_dim, n_features) but pad to embedding_dim if needed
            svd_components = min(self.embedding_dim, n_samples, n_features)
            
            if svd_components < self.embedding_dim:
                print(f"Info: Using {svd_components} SVD components (samples: {n_samples}, features: {n_features}). "
                      f"Output will be padded to {self.embedding_dim} dimensions for FAISS compatibility.")
            
            # Recreate SVD with appropriate number of components
            self.svd = TruncatedSVD(n_components=svd_components, random_state=42)
            
            # Fit SVD
            try:
                self.svd.fit(tfidf_matrix)
                print(f"DEBUG: SVD fit completed successfully")
            except Exception as e:
                print(f"ERROR: SVD fit failed: {e}")
                raise
            
            # Verify fitting was successful immediately after
            vectorizer_fitted_after = (
                hasattr(self.vectorizer, 'vocabulary_') and 
                self.vectorizer.vocabulary_ is not None and
                len(self.vectorizer.vocabulary_) > 0
            )
            svd_fitted_after = (
                hasattr(self.svd, 'components_') and 
                self.svd.components_ is not None and
                self.svd.components_.shape[0] > 0
            )
            print(f"DEBUG: After fitting - vectorizer: {vectorizer_fitted_after}, svd: {svd_fitted_after}, "
                  f"vocab_size: {len(self.vectorizer.vocabulary_) if vectorizer_fitted_after else 0}, "
                  f"svd_components: {self.svd.components_.shape[0] if svd_fitted_after else 0}")
            
            # Double-check using is_fitted method
            is_fitted_check = self.is_fitted()
            print(f"DEBUG: is_fitted() check after _fit_if_needed: {is_fitted_check}")
            
            if not is_fitted_check:
                raise RuntimeError("Model fitting completed but is_fitted() returns False. This should not happen.")
            
            # Save the fitted model to disk
            self._save_model()
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Numpy array of embeddings
        """
        if self.vectorizer is None or self.svd is None:
            self._initialize_model()
        
        # Check if model is fitted using the helper method
        is_fitted = self.is_fitted()
        print(f"DEBUG embed_text: is_fitted={is_fitted}, vectorizer={self.vectorizer is not None}, svd={self.svd is not None}")
        
        if not is_fitted:
            # If model not fitted, we need to fit it first
            # But we can't fit on a single text properly, so raise an error
            raise ValueError(
                "Embedding model has not been fitted yet. Please upload at least one PDF document first "
                "before asking questions. The model needs to be trained on document content."
            )
        
        # Transform - ensure text is not empty
        if not text or not text.strip():
            # Return zero vector if text is empty
            return np.zeros(self.embedding_dim, dtype='float32')
        
        # Transform
        tfidf_vector = self.vectorizer.transform([text])
        
        # Convert sparse matrix to dense if needed and verify shape
        if hasattr(tfidf_vector, 'toarray'):
            tfidf_vector = tfidf_vector.toarray()
        
        # Verify the transform produced a valid shape (should be (1, n_features))
        if len(tfidf_vector.shape) != 2 or tfidf_vector.shape[0] != 1:
            raise ValueError(
                f"TF-IDF transform produced unexpected shape: {tfidf_vector.shape}, "
                f"expected (1, n_features)"
            )
        
        # Check if we have enough features for SVD
        n_features = tfidf_vector.shape[1]
        if n_features < 2:
            raise ValueError(
                f"TF-IDF produced only {n_features} feature(s), but SVD requires at least 2. "
                f"This usually means the question text doesn't match the document vocabulary. "
                f"Please ensure documents are uploaded first and try rephrasing your question."
            )
        
        # Verify SVD expects the same number of features
        if hasattr(self.svd, 'components_'):
            expected_features = self.svd.components_.shape[1]
            if n_features != expected_features:
                raise ValueError(
                    f"Feature dimension mismatch: TF-IDF produced {n_features} features, "
                    f"but SVD expects {expected_features}. This suggests the model was fitted "
                    f"on different data. Please restart the server and upload documents again."
                )
        
        embedding = self.svd.transform(tfidf_vector)
        
        # Pad to fixed embedding dimension if needed
        if embedding.shape[1] < self.embedding_dim:
            padding = np.zeros((embedding.shape[0], self.embedding_dim - embedding.shape[1]))
            embedding = np.hstack([embedding, padding])
        
        # Normalize to unit length for better similarity search
        norm = np.linalg.norm(embedding[0])
        if norm > 0:
            embedding[0] = embedding[0] / norm
        
        return embedding[0].astype('float32')
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            Numpy array of embeddings (shape: [num_texts, embedding_dim])
        """
        if self.vectorizer is None or self.svd is None:
            self._initialize_model()
        
        # Filter out empty texts
        non_empty_texts = [text for text in texts if text and text.strip()]
        if not non_empty_texts:
            raise ValueError("No non-empty texts provided for embedding")
        
        # If we have fewer texts than embedding dimension, we need to adjust
        # But for now, just ensure we have at least one text
        if len(non_empty_texts) < len(texts):
            print(f"Warning: Filtered out {len(texts) - len(non_empty_texts)} empty texts")
        
        # Fit on all texts
        self._fit_if_needed(non_empty_texts)
        
        # Transform all texts (including empty ones - they'll get zero vectors)
        # But we need to handle the case where we filtered texts
        if len(non_empty_texts) < len(texts):
            # Create embeddings for non-empty texts
            tfidf_matrix = self.vectorizer.transform(non_empty_texts)
            embeddings = self.svd.transform(tfidf_matrix)
            
            # Pad to fixed embedding dimension if needed
            if embeddings.shape[1] < self.embedding_dim:
                padding = np.zeros((embeddings.shape[0], self.embedding_dim - embeddings.shape[1]))
                embeddings = np.hstack([embeddings, padding])
            
            # Create zero embeddings for empty texts
            result_embeddings = np.zeros((len(texts), self.embedding_dim), dtype='float32')
            non_empty_idx = 0
            for i, text in enumerate(texts):
                if text and text.strip():
                    result_embeddings[i] = embeddings[non_empty_idx]
                    non_empty_idx += 1
            embeddings = result_embeddings
        else:
            # Transform all texts
            tfidf_matrix = self.vectorizer.transform(texts)
            embeddings = self.svd.transform(tfidf_matrix)
            
            # Pad to fixed embedding dimension if needed
            if embeddings.shape[1] < self.embedding_dim:
                padding = np.zeros((embeddings.shape[0], self.embedding_dim - embeddings.shape[1]))
                embeddings = np.hstack([embeddings, padding])
        
        # Normalize each embedding to unit length
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        embeddings = embeddings / norms
        
        return embeddings.astype('float32')
