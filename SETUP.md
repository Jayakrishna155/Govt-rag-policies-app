# Quick Setup Guide

## Step-by-Step Setup Instructions

### 1. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
echo GROQ_API_KEY=your_groq_api_key_here > .env
# Or manually create .env file with:
# GROQ_API_KEY=your_groq_api_key_here

# Run the serverx
uvicorn main:app --reload --port 8000
```

**Get Groq API Key:**
1. Visit https://console.groq.com/
2. Sign up / Log in
3. Go to API Keys section
4. Create a new API key
5. Copy and paste into `.env` file

### 2. Frontend Setup

```bash
# Navigate to frontend directory (in a new terminal)
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

### 3. Test the Application

1. **Open browser:** http://localhost:5173
2. **Upload a PDF:** Click "Upload PDF" and select a policy document
3. **Wait for processing:** You'll see a success message
4. **Ask a question:** Type a question in the chat interface
5. **View answer:** See the answer with source citations

### 4. Verify Backend API

- **Swagger UI:** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/health

## Troubleshooting

### Backend Issues

**Import Error: `langchain_groq`**
```bash
pip install langchain-groq
```

**FAISS Import Error:**
```bash
# For CPU
pip install faiss-cpu

# For GPU (if available)
pip install faiss-gpu
```

**Module Not Found:**
- Ensure you're in the `backend` directory
- Check that `rag/__init__.py` exists
- Verify Python path includes backend directory

### Frontend Issues

**Port Already in Use:**
- Change port in `vite.config.js` or kill the process using port 5173

**CORS Errors:**
- Ensure backend is running
- Check CORS settings in `backend/main.py`

**API Connection Failed:**
- Verify backend is running on port 8000
- Check `VITE_API_URL` in frontend `.env` (optional)

## First Run Notes

- First PDF upload will take longer (downloading embedding model)
- Embedding model (`all-MiniLM-L6-v2`) is ~80MB and downloads automatically
- FAISS index is created automatically in `backend/data/faiss_index/`
- Document metadata is stored in `backend/data/uploaded_docs.json`

## Production Build

### Backend
```bash
# Use production ASGI server
pip install gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker
```

### Frontend
```bash
cd frontend
npm run build
# Output in frontend/dist/
```

