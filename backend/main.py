"""
FastAPI Main Application
RAG-based Policy Document Question Answering System
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
from dotenv import load_dotenv

from rag.pdf_loader import extract_text_with_pages, extract_text_from_pdf
from rag.duplicate_handler import DuplicateHandler
from rag.text_splitter import chunk_text_with_metadata
from rag.embeddings import EmbeddingModel
from rag.vector_store import FAISSVectorStore
from rag.qa_chain import RAGQAChain

load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Policy Document RAG System",
    description="RAG-based Question Answering System for Policy Documents",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Vite default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
duplicate_handler = DuplicateHandler()
embedding_model = EmbeddingModel()
vector_store = FAISSVectorStore()
qa_chain = RAGQAChain()

# If FAISS has vectors but model isn't fitted, try to refit from metadata
if not embedding_model.is_fitted() and vector_store.get_stats().get('total_vectors', 0) > 0:
    print("Found FAISS index with vectors but no fitted model. Attempting to refit from metadata...")
    metadata = vector_store.metadata
    if metadata:
        texts = [chunk.get('text', '') for chunk in metadata if chunk.get('text', '').strip()]
        if texts:
            success = embedding_model.refit_from_texts(texts)
            if success:
                print("Successfully refitted embedding model from existing FAISS metadata.")
            else:
                print("Failed to refit model from metadata. Please re-upload a PDF.")
        else:
            print("No text found in FAISS metadata. Please re-upload a PDF.")
    else:
        print("No metadata found in FAISS. Please re-upload a PDF.")


class QuestionRequest(BaseModel):
    """Request model for question endpoint"""
    question: str


class QuestionResponse(BaseModel):
    """Response model for question endpoint"""
    answer: str
    sources: List[str]


class UploadResponse(BaseModel):
    """Response model for upload endpoint"""
    status: str
    message: str


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Policy Document RAG System API",
        "version": "1.0.0"
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    stats = vector_store.get_stats()
    return {
        "status": "healthy",
        "vector_store": stats
    }


@app.post("/upload-pdf", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload and process PDF document.
    
    - Checks for duplicates using SHA256 hash
    - Extracts text and chunks it
    - Generates embeddings
    - Updates vector database
    """
    try:
        # Validate file type
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="File must be a PDF")
        
        # Read PDF bytes
        pdf_bytes = await file.read()
        
        if len(pdf_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        
        # Generate document hash
        full_text, document_hash = extract_text_from_pdf(pdf_bytes)
        
        # Check for duplicates
        if duplicate_handler.is_duplicate(document_hash):
            return UploadResponse(
                status="duplicate",
                message=f"Document already exists (hash: {document_hash[:16]}...)"
            )
        
        # Extract text with page numbers
        page_texts = extract_text_with_pages(pdf_bytes)
        
        if not page_texts:
            raise HTTPException(status_code=400, detail="No text extracted from PDF")
        
        # Process each page: chunk and prepare metadata
        all_chunks = []
        for page_text, page_num in page_texts:
            chunks = chunk_text_with_metadata(
                text=page_text,
                page_num=page_num,
                filename=file.filename,
                chunk_size=1000,
                chunk_overlap=200
            )
            all_chunks.extend(chunks)
        
        if not all_chunks:
            raise HTTPException(status_code=400, detail="No chunks created from PDF")
        
        # Generate embeddings for all chunks
        chunk_texts = [chunk['text'] for chunk in all_chunks]
        
        # Filter out empty chunks
        valid_chunks = []
        valid_chunk_texts = []
        for chunk, text in zip(all_chunks, chunk_texts):
            if text and text.strip():
                valid_chunks.append(chunk)
                valid_chunk_texts.append(text)
        
        if not valid_chunk_texts:
            raise HTTPException(status_code=400, detail="No valid text chunks found in PDF")
        
        if len(valid_chunk_texts) < len(chunk_texts):
            print(f"Warning: Filtered out {len(chunk_texts) - len(valid_chunk_texts)} empty chunks")
        
        embeddings = embedding_model.embed_texts(valid_chunk_texts)
        
        # Verify model is now fitted
        is_fitted = embedding_model.is_fitted()
        print(f"DEBUG upload_pdf: Model fitted status after embed_texts: {is_fitted}")
        
        if not is_fitted:
            print("ERROR: Model fitting failed! Check logs above for details.")
            # Try to get more info
            try:
                if hasattr(embedding_model.vectorizer, 'vocabulary_'):
                    vocab = embedding_model.vectorizer.vocabulary_
                    print(f"  Vectorizer vocab exists: {vocab is not None}, size: {len(vocab) if vocab else 0}")
                else:
                    print("  Vectorizer has no vocabulary_ attribute")
            except Exception as e:
                print(f"  Error checking vectorizer: {e}")
            
            try:
                if hasattr(embedding_model.svd, 'components_'):
                    comp = embedding_model.svd.components_
                    print(f"  SVD components exists: {comp is not None}, shape: {comp.shape if comp is not None else 'N/A'}")
                else:
                    print("  SVD has no components_ attribute")
            except Exception as e:
                print(f"  Error checking SVD: {e}")
        else:
            vocab_size = len(embedding_model.vectorizer.vocabulary_) if hasattr(embedding_model.vectorizer, 'vocabulary_') and embedding_model.vectorizer.vocabulary_ is not None else 'unknown'
            print(f"SUCCESS: Model fitted. Vocabulary size: {vocab_size}")
        
        # Update all_chunks to only include valid chunks
        all_chunks = valid_chunks
        
        # Add to vector store
        vector_store.add_vectors(embeddings, all_chunks)
        
        # Mark document as processed
        duplicate_handler.add_document(document_hash, file.filename)
        
        return UploadResponse(
            status="success",
            message=f"PDF processed and stored. Added {len(all_chunks)} chunks."
        )
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error processing PDF: {str(e)}")
        print(f"Traceback: {error_details}")
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Answer question using RAG pipeline.
    
    - Generates query embedding
    - Searches vector database
    - Retrieves relevant chunks
    - Generates answer using Groq LLM
    """
    try:
        question = request.question.strip()
        
        if not question:
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        # Check if model is fitted, and if not, check if we have vectors in FAISS
        if not embedding_model.is_fitted():
            stats = vector_store.get_stats()
            if stats.get('total_vectors', 0) > 0:
                raise HTTPException(
                    status_code=500,
                    detail="Embedding model state was lost. Please restart the server and re-upload your PDF documents, "
                           "or the model will be automatically refitted on the next upload."
                )
        
        # Generate query embedding
        query_embedding = embedding_model.embed_text(question)
        
        # Search vector store - use more chunks for summarization requests
        is_summarization = any(word in question.lower() for word in ['summarize', 'summary', 'overview', 'summarise'])
        k = 8 if is_summarization else 4  # More chunks for better summarization
        retrieved_chunks = vector_store.search(query_embedding, k=k)
        
        # Generate answer using RAG chain
        result = qa_chain.answer_question(question, retrieved_chunks)
        
        return QuestionResponse(
            answer=result["answer"],
            sources=result["sources"]
        )
    
    except HTTPException:
        raise
    except ValueError as e:
        # Handle specific embedding errors with helpful messages
        error_msg = str(e)
        if "not been fitted" in error_msg or "upload" in error_msg.lower():
            raise HTTPException(
                status_code=400, 
                detail="Please upload at least one PDF document before asking questions. The embedding model needs to be trained on document content first."
            )
        raise HTTPException(status_code=500, detail=f"Error answering question: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error answering question: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

