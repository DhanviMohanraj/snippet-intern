import React from 'react';
import { Plus, Trash2, MessageSquare, User, Layers } from 'lucide-react';

export default function Sidebar({
  conversations = [],
  activeConversationId,
  onSelectConversation,
  onNewConversation,
  onDeleteConversation,
  userId,
  isOpen,
}) {
  const handleDelete = (e, convId) => {
    e.stopPropagation();
    if (window.confirm('Are you sure you want to delete this consultation session?')) {
      onDeleteConversation(convId);
    }
  };

  return (
    <aside className={`app-sidebar ${isOpen ? 'open' : ''}`}>
      <div className="sidebar-header">
        <span className="sidebar-title">Consultations</span>
        <button 
          className="header-btn" 
          onClick={onNewConversation} 
          title="New consultation"
          aria-label="New consultation"
        >
          <Plus size={14} />
        </button>
      </div>

      <div className="sidebar-content">
        {conversations.length === 0 ? (
          <div style={{ padding: '1rem', color: 'var(--text-muted)', fontSize: '0.82rem', textAlign: 'center' }}>
            <Layers size={24} style={{ margin: '0 auto 0.5rem', opacity: 0.4 }} />
            <p>No active sessions.</p>
            <p style={{ fontSize: '0.75rem', marginTop: '0.25rem' }}>Start a consultation to find electronics.</p>
          </div>
        ) : (
          conversations.map((conv) => {
            const isActive = conv.id === activeConversationId;
            const title = conv.preview || `Session ${conv.id.substring(0, 8)}...`;
            const dateStr = conv.updated_at 
              ? new Date(conv.updated_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
              : 'Active';

            return (
              <div
                key={conv.id}
                className={`conversation-item ${isActive ? 'active' : ''}`}
                onClick={() => onSelectConversation(conv.id)}
                role="button"
                tabIndex={0}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') onSelectConversation(conv.id);
                }}
              >
                <div className="conversation-item-info">
                  <span className="conversation-item-title">{title}</span>
                  <span className="conversation-item-meta">
                    {conv.turn_count ? `${conv.turn_count} turns • ` : ''}
                    {dateStr}
                  </span>
                </div>
                <button
                  className="delete-session-btn"
                  onClick={(e) => handleDelete(e, conv.id)}
                  title="Delete consultation"
                  aria-label={`Delete consultation ${conv.id}`}
                >
                  <Trash2 size={13} />
                </button>
              </div>
            );
          })
        )}
      </div>

      <div className="sidebar-footer">
        <div className="user-id-pill" title="User ID sent in X-User-ID header">
          <User size={13} />
          <span>User: {userId ? userId.substring(0, 16) : 'anonymous'}</span>
        </div>
      </div>
    </aside>
  );
}
