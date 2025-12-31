import React, { useState, useRef, useEffect } from 'react';
import { askQuestion } from '../services/api';
import './ChatBot.css';

const ChatBot = () => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    const question = inputValue.trim();
    if (!question || loading) return;

    // Add user message
    const userMessage = {
      type: 'user',
      content: question,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputValue('');
    setLoading(true);

    try {
      const response = await askQuestion(question);
      
      const aiMessage = {
        type: 'ai',
        content: response.answer,
        sources: response.sources || [],
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, aiMessage]);
    } catch (err) {
      const errorMessage = {
        type: 'ai',
        content: err.response?.data?.detail || 
                 err.message || 
                 'Error getting response. Please try again.',
        sources: [],
        timestamp: new Date(),
        isError: true,
      };

      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleClear = () => {
    setMessages([]);
  };

  return (
    <div className="chatbot-container">
      <div className="chatbot-card">
        <div className="chatbot-header">
          <h2>Ask Questions</h2>
          <p className="chatbot-subtitle">
            Ask questions about the uploaded policy documents
          </p>
          {messages.length > 0 && (
            <button onClick={handleClear} className="clear-button">
              Clear Conversation
            </button>
          )}
        </div>

        <div className="messages-container">
          {messages.length === 0 ? (
            <div className="empty-state">
              <div className="empty-icon">💬</div>
              <p>Start a conversation by asking a question</p>
              <p className="empty-hint">
                Example: "What is the policy on remote work?"
              </p>
            </div>
          ) : (
            messages.map((message, index) => (
              <div
                key={index}
                className={`message ${message.type} ${message.isError ? 'error' : ''}`}
              >
                <div className="message-content">
                  {message.type === 'user' ? (
                    <div className="message-bubble user-bubble">
                      {message.content}
                    </div>
                  ) : (
                    <div className="message-bubble ai-bubble">
                      <div className="ai-response">{message.content}</div>
                      {message.sources && message.sources.length > 0 && (
                        <div className="sources">
                          <strong>Sources:</strong>
                          <ul>
                            {message.sources.map((source, idx) => (
                              <li key={idx}>{source}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>
            ))
          )}

          {loading && (
            <div className="message ai">
              <div className="message-bubble ai-bubble">
                <div className="loading-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        <div className="input-container">
          <textarea
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Type your question here..."
            className="chat-input"
            rows="2"
            disabled={loading}
          />
          <button
            onClick={handleSend}
            disabled={!inputValue.trim() || loading}
            className="send-button"
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
};

export default ChatBot;

