import React, { useState } from 'react';
import { uploadPDF } from '../services/api';
import './UploadPDF.css';

const UploadPDF = ({ onUploadSuccess }) => {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [status, setStatus] = useState(null);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      if (selectedFile.type !== 'application/pdf') {
        setError('Please select a PDF file');
        setFile(null);
        return;
      }
      setFile(selectedFile);
      setError(null);
      setStatus(null);
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a file first');
      return;
    }

    setUploading(true);
    setError(null);
    setStatus(null);

    try {
      const response = await uploadPDF(file);
      
      if (response.status === 'success') {
        setStatus({
          type: 'success',
          message: response.message,
        });
        setFile(null);
        // Reset file input
        document.getElementById('pdf-upload-input').value = '';
        if (onUploadSuccess) {
          onUploadSuccess();
        }
      } else if (response.status === 'duplicate') {
        setStatus({
          type: 'warning',
          message: response.message,
        });
        setFile(null);
        document.getElementById('pdf-upload-input').value = '';
      }
    } catch (err) {
      setError(
        err.response?.data?.detail || 
        err.message || 
        'Error uploading PDF. Please try again.'
      );
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="upload-container">
      <div className="upload-card">
        <h2>Upload Policy Document</h2>
        <p className="upload-description">
          Upload a PDF document to add it to the knowledge base
        </p>

        <div className="upload-input-group">
          <input
            id="pdf-upload-input"
            type="file"
            accept=".pdf"
            onChange={handleFileChange}
            disabled={uploading}
            className="file-input"
          />
          <button
            onClick={handleUpload}
            disabled={!file || uploading}
            className="upload-button"
          >
            {uploading ? 'Uploading...' : 'Upload PDF'}
          </button>
        </div>

        {file && (
          <div className="file-info">
            <span className="file-name">📄 {file.name}</span>
            <span className="file-size">
              {(file.size / 1024 / 1024).toFixed(2)} MB
            </span>
          </div>
        )}

        {status && (
          <div className={`status-message status-${status.type}`}>
            {status.type === 'success' ? '✅' : '⚠️'} {status.message}
          </div>
        )}

        {error && (
          <div className="status-message status-error">
            ❌ {error}
          </div>
        )}
      </div>
    </div>
  );
};

export default UploadPDF;

