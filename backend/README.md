# Policy Document RAG System - Backend

## Setup Instructions

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables:**
   - Copy `.env.example` to `.env`
   - Add your Groq API key:
     ```
     GROQ_API_KEY=your_groq_api_key_here
     ```
   - Get your API key from: https://console.groq.com/

3. **Run the server:**
   ```bash
   uvicorn main:app --reload --port 8000
   ```

4. **API Documentation:**
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

## API Endpoints

### POST `/upload-pdf`
Upload a PDF document for processing.

**Request:** Multipart form data with PDF file

**Response:**
```json
{
  "status": "success",
  "message": "PDF processed and stored. Added 45 chunks."
}
```

### POST `/ask`
Ask a question about uploaded documents.

**Request:**
```json
{
  "question": "What is the policy on remote work?"
}
```

**Response:**
```json
{
  "answer": "According to the policy document...",
  "sources": ["Page 3 of policy.pdf", "Page 7 of policy.pdf"]
}
```

### GET `/health`
Check system health and vector store statistics.

## Project Structure

```
backend/
├── main.py                 # FastAPI application
├── rag/
│   ├── pdf_loader.py      # PDF text extraction
│   ├── duplicate_handler.py  # Duplicate detection
│   ├── text_splitter.py   # Text chunking
│   ├── embeddings.py      # Hugging Face embeddings
│   ├── vector_store.py    # FAISS vector database
│   └── qa_chain.py        # RAG QA chain with Groq
├── data/
│   ├── faiss_index/       # FAISS index files
│   └── uploaded_docs.json # Document metadata
└── requirements.txt
```

