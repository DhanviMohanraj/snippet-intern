import React from 'react';
import { 
  Bot, 
  User, 
  HelpCircle, 
  AlertOctagon, 
  Sparkles 
} from 'lucide-react';
import RecommendationCard from './RecommendationCard.jsx';

export default function MessageBubble({ message, onAskFollowUp }) {
  const isUser = message.role === 'user';
  const metadata = message.metadata || {};
  const requiresClarification = message.requires_clarification || metadata.requires_clarification;
  const clarificationQuestion = message.clarification_question || metadata.clarification_question;
  const contradictions = message.unresolved_contradictions || metadata.unresolved_contradictions || [];
  const recommendations = message.recommendations || metadata.recommendations || [];
  const action = message.action || metadata.action;
  const turnIndex = message.turn_index ?? message.turn_count;

  const comparisonSummary = message.comparison_summary || metadata.comparison_summary;
  const remainingUncertainties = message.remaining_uncertainties || metadata.remaining_uncertainties || [];

  return (
    <div className={`message-row ${isUser ? 'user' : 'assistant'}`}>
      {!isUser && (
        <div className="message-avatar assistant" title="SpecSage Assistant">
          <Bot size={18} />
        </div>
      )}

      <div className={`message-bubble ${isUser ? 'user' : 'assistant'}`}>
        <div className="message-header">
          <span>{isUser ? 'You' : 'SpecSage'}</span>
          {turnIndex != null && (
            <span className="message-turn-tag">Turn {turnIndex}</span>
          )}
        </div>

        <div className="message-text">
          {message.content || message.message || ''}
        </div>

        {/* Clarification Callout */}
        {requiresClarification && clarificationQuestion && clarificationQuestion.trim() !== (message.content || message.message || '').trim() && (
          <div className="clarification-callout" role="status">
            <div className="clarification-callout-header">
              <HelpCircle size={14} />
              <span>Clarification Needed</span>
            </div>
            <div className="clarification-callout-question">
              {clarificationQuestion}
            </div>
          </div>
        )}

        {/* Contradiction Callout */}
        {contradictions.length > 0 && (
          <div className="contradiction-callout" role="alert">
            <div className="contradiction-header">
              <AlertOctagon size={14} />
              <span>Requirement Conflict Detected</span>
            </div>
            {contradictions.map((con, idx) => (
              <p key={idx} style={{ fontSize: '0.8rem', color: '#fca5a5' }}>
                {typeof con === 'string' ? con : con.description || con.issue || JSON.stringify(con)}
              </p>
            ))}
          </div>
        )}

        {/* Product Recommendations */}
        {recommendations.length > 0 && (
          <div className="recommendations-container">
            <div className="recommendations-heading">
              <Sparkles size={16} />
              <span>Tailored Recommendations ({recommendations.length})</span>
            </div>
            {recommendations.map((rec, idx) => (
              <RecommendationCard
                key={rec.product_id}
                recommendation={rec}
                rank={idx + 1}
                onAskFollowUp={onAskFollowUp}
              />
            ))}
          </div>
        )}

        {/* Direct Comparison Summary */}
        {comparisonSummary && (
          <div
            className="comparison-callout"
            style={{
              background: 'rgba(99, 102, 241, 0.08)',
              border: '1px solid rgba(99, 102, 241, 0.25)',
              borderRadius: 'var(--radius-md)',
              padding: '0.75rem 1rem',
              marginTop: '0.75rem',
            }}
          >
            <div style={{ fontSize: '0.8rem', fontWeight: 700, color: '#a5b4fc', marginBottom: '0.3rem' }}>
              Comparison Summary
            </div>
            <p style={{ fontSize: '0.84rem', color: 'var(--text-secondary)', lineHeight: 1.45, margin: 0 }}>
              {comparisonSummary}
            </p>
          </div>
        )}

        {/* Remaining Uncertainties Callout */}
        {remainingUncertainties.length > 0 && (
          <div
            className="uncertainties-callout"
            style={{
              background: 'rgba(234, 179, 8, 0.06)',
              border: '1px solid rgba(234, 179, 8, 0.2)',
              borderRadius: 'var(--radius-md)',
              padding: '0.65rem 0.85rem',
              marginTop: '0.6rem',
            }}
          >
            <div style={{ fontSize: '0.75rem', fontWeight: 700, color: '#fbbf24', textTransform: 'uppercase', marginBottom: '0.25rem' }}>
              Remaining Catalogue Uncertainties
            </div>
            {remainingUncertainties.map((unc, idx) => (
              <div key={idx} style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', lineHeight: 1.35, display: 'flex', gap: '0.35rem' }}>
                <span>•</span>
                <span>{unc}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      {isUser && (
        <div className="message-avatar user" title="You">
          <User size={18} />
        </div>
      )}
    </div>
  );
}
