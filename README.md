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

