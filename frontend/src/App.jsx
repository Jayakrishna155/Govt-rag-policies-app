import React, { useState } from 'react';
import UploadPDF from './components/UploadPDF';
import ChatBot from './components/ChatBot';
import './App.css';

function App() {
  const [uploadKey, setUploadKey] = useState(0);

  const handleUploadSuccess = () => {
    // Force re-render to show updated state
    setUploadKey((prev) => prev + 1);
  };

  return (
    <div className="app">
      <div className="app-container">
        <header className="app-header">
          <h1>📚 Policy Document RAG System</h1>
          <p className="app-subtitle">
            Upload policy documents and ask questions using AI-powered RAG
          </p>
        </header>

        <main className="app-main">
          <UploadPDF key={uploadKey} onUploadSuccess={handleUploadSuccess} />
          <ChatBot />
        </main>

        <footer className="app-footer">
          <p>
            Built with React, FastAPI, LangChain, Hugging Face, and Groq
          </p>
        </footer>
      </div>
    </div>
  );
}

export default App;

