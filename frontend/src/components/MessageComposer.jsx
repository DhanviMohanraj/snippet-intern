import React, { useState, useRef, useEffect } from 'react';
import { Send, Loader2, AlertCircle, RotateCcw } from 'lucide-react';

export default function MessageComposer({
  onSendMessage,
  isLoading = false,
  errorMessage = null,
  onRetry = null,
  initialText = '',
}) {
  const [inputText, setInputText] = useState(initialText || '');
  const textareaRef = useRef(null);

  useEffect(() => {
    if (initialText) {
      setInputText(initialText);
      if (textareaRef.current) {
        textareaRef.current.focus();
      }
    }
  }, [initialText]);

  // Adjust textarea height automatically
  const handleInputChange = (e) => {
    setInputText(e.target.value);
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = `${Math.min(textarea.scrollHeight, 140)}px`;
    }
  };

  const handleSubmit = (e) => {
    if (e) e.preventDefault();
    if (!inputText.trim() || isLoading) return;

    const textToSend = inputText.trim();
    onSendMessage(textToSend);
    setInputText('');

    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <footer className="composer-area">
      {errorMessage && (
        <div className="composer-error-alert" role="alert">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <AlertCircle size={15} />
            <span>{errorMessage}</span>
          </div>
          {onRetry && (
            <button
              type="button"
              className="retry-action-btn"
              onClick={onRetry}
              title="Retry sending message"
            >
              <RotateCcw size={12} style={{ display: 'inline', marginRight: 4 }} />
              Retry
            </button>
          )}
        </div>
      )}

      <form className="composer-box" onSubmit={handleSubmit}>
        <textarea
          ref={textareaRef}
          className="composer-textarea"
          placeholder="Describe what you need (e.g., '16GB RAM laptop for machine learning under ₹85,000')..."
          rows={1}
          value={inputText}
          onChange={handleInputChange}
          onKeyDown={handleKeyDown}
          disabled={isLoading}
          aria-label="Message prompt input"
        />

        <button
          type="submit"
          className="send-btn"
          disabled={!inputText.trim() || isLoading}
          title="Send message"
          aria-label="Send message"
        >
          {isLoading ? (
            <Loader2 size={16} className="spinner" />
          ) : (
            <Send size={16} />
          )}
        </button>
      </form>

      <div className="composer-footer-hints">
        <span>Press <strong>Enter</strong> to send, <strong>Shift + Enter</strong> for new line</span>
        <span>SpecSage Multi-turn Conversational AI</span>
      </div>
    </footer>
  );
}
