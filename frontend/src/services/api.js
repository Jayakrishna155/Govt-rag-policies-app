/**
 * API Service Layer
 * Handles all backend API calls
 */
import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

/**
 * Upload PDF file
 * @param {File} file - PDF file to upload
 * @returns {Promise} API response
 */
export const uploadPDF = async (file) => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await api.post('/upload-pdf', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });

  return response.data;
};

/**
 * Ask a question
 * @param {string} question - User's question
 * @returns {Promise} API response with answer and sources
 */
export const askQuestion = async (question) => {
  const response = await api.post('/ask', {
    question: question,
  });

  return response.data;
};

/**
 * Check system health
 * @returns {Promise} API response with system stats
 */
export const checkHealth = async () => {
  const response = await api.get('/health');
  return response.data;
};

