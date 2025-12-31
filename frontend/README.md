# Policy Document RAG System - Frontend

## Setup Instructions

1. **Install dependencies:**
   ```bash
   npm install
   ```

2. **Set up environment variables (optional):**
   Create a `.env` file in the frontend directory:
   ```
   VITE_API_URL=http://localhost:8000
   ```
   (Defaults to `http://localhost:8000` if not set)

3. **Run the development server:**
   ```bash
   npm run dev
   ```

4. **Build for production:**
   ```bash
   npm run build
   ```

5. **Preview production build:**
   ```bash
   npm run preview
   ```

## Features

- **PDF Upload**: Upload policy documents with duplicate detection
- **Chat Interface**: Ask questions about uploaded documents
- **Source Citations**: View page numbers and filenames for answers
- **Real-time Status**: Upload status and error handling
- **Responsive Design**: Works on desktop and mobile devices

## Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── UploadPDF.jsx    # PDF upload component
│   │   ├── UploadPDF.css
│   │   ├── ChatBot.jsx      # Chat interface component
│   │   └── ChatBot.css
│   ├── services/
│   │   └── api.js           # API service layer
│   ├── App.jsx              # Main app component
│   ├── App.css
│   ├── main.jsx             # Entry point
│   └── index.css            # Global styles
├── index.html
├── package.json
└── vite.config.js
```

