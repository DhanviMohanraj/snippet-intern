import React, { useEffect, useRef } from 'react';
import { Sparkles, Laptop, Smartphone, Tablet, Headphones, Bot } from 'lucide-react';
import MessageBubble from './MessageBubble.jsx';

const STARTER_PROMPTS = [
  {
    icon: <Laptop size={18} color="#818cf8" />,
    title: 'College Coding Laptop',
    prompt: 'I need a laptop for college programming, Python development, and VS Code under ₹70,000',
  },
  {
    icon: <Laptop size={18} color="#34d399" />,
    title: 'High-Performance Gaming',
    prompt: 'Looking for a gaming laptop with RTX 4060, 16GB RAM, and 144Hz display under ₹1,20,000',
  },
  {
    icon: <Smartphone size={18} color="#38bdf8" />,
    title: 'Flagship Camera Phone',
    prompt: 'Need a flagship smartphone with top-tier camera and 256GB storage under ₹80,000',
  },
  {
    icon: <Tablet size={18} color="#fbbf24" />,
    title: 'Student Study Tablet',
    prompt: 'Looking for a tablet with stylus support for taking lecture notes and reading PDFs under ₹35,000',
  },
];

export default function ChatArea({
  messages = [],
  isLoading = false,
  onSelectStarterPrompt,
  onAskFollowUp,
}) {
  const messagesEndRef = useRef(null);

  useEffect(() => {
    if (typeof messagesEndRef.current?.scrollIntoView === 'function') {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, isLoading]);

  return (
    <div className="chat-stream-column" role="log" aria-label="Conversation history">
      {messages.length === 0 ? (
        <div className="chat-hero">
          <div className="hero-icon-container">
            <Sparkles size={32} />
          </div>
          <h2 className="hero-heading">What device are you looking for?</h2>
          <p className="hero-subheading">
            Tell me your budget, intended use cases, and preferred specs. I will converse with you 
            to understand your exact requirements before recommending matching products.
          </p>

          <div className="starter-prompts-grid">
            {STARTER_PROMPTS.map((item, idx) => (
              <div
                key={idx}
                className="starter-prompt-card"
                onClick={() => onSelectStarterPrompt(item.prompt)}
                role="button"
                tabIndex={0}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') onSelectStarterPrompt(item.prompt);
                }}
              >
                <div className="starter-prompt-title">
                  {item.icon}
                  <span>{item.title}</span>
                </div>
                <div className="starter-prompt-text">{item.prompt}</div>
              </div>
            ))}
          </div>
        </div>
      ) : (
        messages.map((msg, index) => (
          <MessageBubble
            key={msg.id || index}
            message={msg}
            onAskFollowUp={onAskFollowUp}
          />
        ))
      )}

      {isLoading && (
        <div className="typing-row">
          <div className="message-avatar assistant" title="SpecSage Assistant">
            <Bot size={18} />
          </div>
          <div className="typing-bubble">
            <div className="typing-dots">
              <span className="typing-dot" />
              <span className="typing-dot" />
              <span className="typing-dot" />
            </div>
            <span className="typing-label">Analyzing specifications...</span>
          </div>
        </div>
      )}

      <div ref={messagesEndRef} />
    </div>
  );
}
