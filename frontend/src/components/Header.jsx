import React from 'react';
import { 
  Cpu, 
  MessageSquare, 
  ShoppingBag, 
  Plus, 
  Menu, 
  SlidersHorizontal,
  RefreshCw 
} from 'lucide-react';

export default function Header({
  activeTab,
  onTabChange,
  onNewConsultation,
  healthStatus,
  onRefreshHealth,
  sidebarOpen,
  onToggleSidebar,
  requirementsOpen,
  onToggleRequirements,
}) {
  const getHealthBadge = () => {
    if (!healthStatus) {
      return (
        <div className="health-badge" title="Checking backend health...">
          <span className="health-dot warning" />
          <span>Connecting...</span>
        </div>
      );
    }

    const isDbOk = healthStatus.database?.status === 'ok';
    const isLlmOk = healthStatus.llm?.status === 'ok';

    if (isDbOk && isLlmOk) {
      return (
        <div className="health-badge" title="MongoDB and LLM services operational">
          <span className="health-dot ok" />
          <span>System Online</span>
        </div>
      );
    }

    return (
      <div 
        className="health-badge" 
        title={`DB: ${healthStatus.database?.status || 'unknown'}, LLM: ${healthStatus.llm?.status || 'unknown'}`}
      >
        <span className="health-dot error" />
        <span>Degraded</span>
      </div>
    );
  };

  return (
    <header className="app-header" role="banner">
      <div className="header-left">
        <button 
          className="header-btn" 
          onClick={onToggleSidebar} 
          title="Toggle Consultation Sessions"
          aria-label="Toggle Sessions"
        >
          <Menu size={18} />
        </button>

        <div className="brand-badge">
          <div className="brand-logo-icon">
            <Cpu size={20} />
          </div>
          <div>
            <span className="brand-title">SpecSage</span>
            <span className="brand-tag">AI Advisor</span>
          </div>
        </div>
      </div>

      <div className="header-center">
        <button
          className={`nav-tab-btn ${activeTab === 'chat' ? 'active' : ''}`}
          onClick={() => onTabChange('chat')}
          aria-label="AI Consultation Mode"
        >
          <MessageSquare size={16} />
          <span>AI Consultation</span>
        </button>
        <button
          className={`nav-tab-btn ${activeTab === 'catalogue' ? 'active' : ''}`}
          onClick={() => onTabChange('catalogue')}
          aria-label="Catalogue Browser Mode"
        >
          <ShoppingBag size={16} />
          <span>Product Catalogue</span>
        </button>
      </div>

      <div className="header-right">
        {getHealthBadge()}

        <button 
          className="header-btn" 
          onClick={onRefreshHealth} 
          title="Refresh health status"
          aria-label="Refresh status"
        >
          <RefreshCw size={14} />
        </button>

        {activeTab === 'chat' && (
          <>
            <button 
              className="header-btn primary" 
              onClick={onNewConsultation} 
              title="Start a new consultation session"
            >
              <Plus size={16} />
              <span>New Consultation</span>
            </button>

            <button 
              className={`header-btn ${requirementsOpen ? 'active' : ''}`}
              onClick={onToggleRequirements} 
              title="Toggle Live Specs Requirements Panel"
              aria-label="Toggle Requirements"
            >
              <SlidersHorizontal size={16} />
            </button>
          </>
        )}
      </div>
    </header>
  );
}
