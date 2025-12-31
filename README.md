# Policy Document Question Answering System using RAG

A full-stack RAG (Retrieval-Augmented Generation) application for question answering over policy documents. Built as a solo project showcasing modern AI/ML engineering practices.

## 🎯 Project Overview

This application enables users to:
- Upload policy PDF documents
- Automatically detect and prevent duplicate uploads
- Ask questions about the uploaded documents
- Get accurate answers with source citations (page numbers)

## 🏗️ Tech Stack

### Frontend
- **React.js** - UI framework
- **Vite** - Build tool
- **Axios** - HTTP client

### Backend
- **Python** - Programming language
- **FastAPI** - Web framework
- **LangChain** - RAG pipeline orchestration
- **FAISS** - Vector database (local, persistent)
- **Hugging Face** - Sentence transformers for embeddings (`all-MiniLM-L6-v2`)
- **Groq** - LLM provider (Llama 3.1 70B)

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- Groq API key (get from https://console.groq.com/)

### Backend Setup

1. **Navigate to backend directory:**
   ```bash
   cd backend
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   ```bash
   # Create .env file
   echo "GROQ_API_KEY=your_groq_api_key_here" > .env
   ```

4. **Run the server:**
   ```bash
   uvicorn main:app --reload --port 8000
   ```

   API will be available at: http://localhost:8000
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

### Frontend Setup

1. **Navigate to frontend directory:**
   ```bash
   cd frontend
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Run the development server:**
   ```bash
   npm run dev
   ```

   Frontend will be available at: http://localhost:5173

## 📂 Project Structure

```
.
├── backend/
│   ├── main.py                    # FastAPI application
│   ├── requirements.txt           # Python dependencies
│   ├── .env                       # Environment variables
│   ├── rag/
│   │   ├── pdf_loader.py         # PDF text extraction
│   │   ├── duplicate_handler.py  # Duplicate detection (SHA256)
│   │   ├── text_splitter.py      # Text chunking
│   │   ├── embeddings.py         # Hugging Face embeddings
│   │   ├── vector_store.py       # FAISS vector database
│   │   └── qa_chain.py           # RAG QA chain with Groq
│   └── data/
│       ├── faiss_index/           # FAISS index files
│       └── uploaded_docs.json     # Document metadata
│
└── frontend/
    ├── src/
    │   ├── components/
    │   │   ├── UploadPDF.jsx     # PDF upload component
    │   │   └── ChatBot.jsx       # Chat interface
    │   ├── services/
    │   │   └── api.js            # API service layer
    │   ├── App.jsx               # Main app component
    │   └── main.jsx              # Entry point
    ├── package.json
    └── vite.config.js
```

## 🔄 Application Flow

### 1. PDF Upload Flow
1. User uploads PDF via frontend
2. Backend extracts text and generates SHA256 hash
3. Duplicate check: if hash exists, skip processing
4. If new: chunk text (1000 chars, 200 overlap)
5. Generate embeddings using Hugging Face model
6. Add vectors to FAISS index with metadata
7. Save index and metadata to disk

### 2. Question Answering Flow
1. User asks a question
2. Generate query embedding
3. Search FAISS for top-4 similar chunks
4. Retrieve context from chunks
5. Pass context + question to Groq LLM
6. LLM generates answer from context only
7. Return answer with source citations

## 🔌 API Endpoints

### `POST /upload-pdf`
Upload a PDF document.

**Request:** Multipart form data with PDF file

**Response:**
```json
{
  "status": "success",
  "message": "PDF processed and stored. Added 45 chunks."
}
```

### `POST /ask`
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

### `GET /health`
Check system health and vector store statistics.

## 🎯 Key Features

- ✅ **Duplicate Prevention**: SHA256 hashing prevents re-processing same documents
- ✅ **Dynamic Vector DB**: FAISS index created/updated automatically
- ✅ **Persistent Storage**: Vector index and metadata saved to disk
- ✅ **Source Citations**: Answers include page numbers and filenames
- ✅ **Context-Only Answers**: LLM instructed to answer only from uploaded documents
- ✅ **Clean UI**: Modern, responsive design
- ✅ **Error Handling**: Comprehensive error handling throughout

## 🧪 Testing

1. **Upload a PDF:**
   - Use the upload interface
   - Check for success message
   - Try uploading the same file again (should detect duplicate)

2. **Ask Questions:**
   - Ask questions relevant to uploaded documents
   - Verify answers include source citations
   - Ask questions not in documents (should return "Answer not found")

3. **Check Health:**
   - Visit http://localhost:8000/health
   - Verify vector store statistics

## 📝 Resume Description

> **Built a full-stack RAG system using React, FastAPI, LangChain, Hugging Face embeddings, and Groq LLM to enable accurate question answering over uploaded policy documents with dynamic vector database management and duplicate handling.**

## 🔧 Troubleshooting

### Backend Issues
- **Import errors**: Ensure all dependencies are installed (`pip install -r requirements.txt`)
- **Groq API errors**: Verify `GROQ_API_KEY` is set correctly in `.env`
- **FAISS errors**: Ensure `faiss-cpu` is installed (or `faiss-gpu` if using GPU)

### Frontend Issues
- **CORS errors**: Ensure backend CORS middleware allows frontend origin
- **API connection errors**: Verify backend is running on port 8000
- **Build errors**: Clear `node_modules` and reinstall (`rm -rf node_modules && npm install`)

## 📄 License

This project is built as a portfolio/resume project.

## 🙏 Acknowledgments

- LangChain for RAG pipeline tools
- Hugging Face for embeddings
- Groq for fast LLM inference
- FAISS for efficient vector search

