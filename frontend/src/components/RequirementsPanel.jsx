import React from 'react';
import { 
  SlidersHorizontal, 
  CheckCircle2, 
  Clock, 
  AlertTriangle, 
  DollarSign, 
  Laptop, 
  Smartphone, 
  Tablet, 
  Headphones, 
  Monitor, 
  Sparkles 
} from 'lucide-react';

export default function RequirementsPanel({
  requirements = null,
  canRecommend = false,
  remainingSlots = [],
  slotStatus = {},
  contradictions = [],
  onSelectSlotHint,
  isOpen = true,
}) {
  if (!isOpen) return null;

  const getCategoryIcon = (cat) => {
    const c = (cat || '').toLowerCase();
    if (c.includes('laptop')) return <Laptop size={14} />;
    if (c.includes('phone') || c.includes('mobile')) return <Smartphone size={14} />;
    if (c.includes('tablet')) return <Tablet size={14} />;
    if (c.includes('headphone') || c.includes('audio')) return <Headphones size={14} />;
    return <Monitor size={14} />;
  };

  const formatPrice = (val) => {
    if (!val) return 'Flexible';
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR',
      maximumFractionDigits: 0,
    }).format(val);
  };

  // Convert slotStatus or hard constraints into viewable items
  const hardConstraints = requirements?.hard || {};
  const category = requirements?.category || null;
  const budget = requirements?.budget_max_inr || hardConstraints?.budget_max_inr || null;
  const useCases = requirements?.use_cases || [];

  const slotEntries = Object.entries(slotStatus || {});

  return (
    <aside className="requirements-panel" aria-label="Requirements Status Panel">
      <div className="requirements-panel-header">
        <span className="panel-title">
          <SlidersHorizontal size={15} />
          Requirements Spec
        </span>
        <span className={`status-badge ${canRecommend ? 'ready' : 'gathering'}`}>
          {canRecommend ? 'Ready to Recommend' : 'Gathering Specs'}
        </span>
      </div>

      {contradictions && contradictions.length > 0 && (
        <div className="contradiction-callout" role="alert">
          <div className="contradiction-header">
            <AlertTriangle size={14} />
            <span>Specification Conflict</span>
          </div>
          {contradictions.map((c, i) => (
            <p key={i} style={{ fontSize: '0.78rem', color: '#fca5a5' }}>
              {c.description || c.issue || 'Conflicting requirements detected.'}
            </p>
          ))}
        </div>
      )}

      {/* Target Category & Budget */}
      <div className="req-section">
        <span className="req-section-title">Target Parameters</span>
        <div className="req-card">
          <div className="req-item-row">
            <span className="req-item-label">Device Type:</span>
            <span className="req-item-value" style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
              {category ? (
                <>
                  {getCategoryIcon(category)}
                  <span style={{ textTransform: 'capitalize' }}>{category}</span>
                </>
              ) : (
                <span style={{ color: 'var(--text-muted)' }}>Not identified</span>
              )}
            </span>
          </div>

          <div className="req-item-row">
            <span className="req-item-label">Max Budget:</span>
            <span className="req-item-value">
              {budget ? formatPrice(budget) : <span style={{ color: 'var(--text-muted)' }}>Not specified</span>}
            </span>
          </div>

          {useCases.length > 0 && (
            <div className="req-item-row" style={{ alignItems: 'flex-start', flexDirection: 'column', gap: '0.2rem' }}>
              <span className="req-item-label">Primary Use Cases:</span>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.25rem', marginTop: '0.15rem' }}>
                {useCases.map((uc, i) => (
                  <span key={i} className="spec-pill" style={{ fontSize: '0.7rem' }}>
                    {uc}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Slot Status Engine */}
      <div className="req-section">
        <span className="req-section-title">Decision Slots</span>
        <div className="slots-grid">
          {slotEntries.length > 0 ? (
            slotEntries.map(([slotKey, status]) => {
              const isAnswered = status === 'answered' || status === 'satisfied';
              return (
                <div 
                  key={slotKey} 
                  className={`slot-item ${isAnswered ? 'answered' : 'pending'}`}
                >
                  <span className="slot-name">
                    {isAnswered ? (
                      <CheckCircle2 size={13} color="var(--accent-success)" />
                    ) : (
                      <Clock size={13} color="var(--accent-warning)" />
                    )}
                    {slotKey.replace(/_/g, ' ')}
                  </span>
                  <span className={`slot-state-tag ${isAnswered ? 'answered' : 'pending'}`}>
                    {isAnswered ? 'Answered' : 'Pending'}
                  </span>
                </div>
              );
            })
          ) : (
            <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', padding: '0.5rem 0' }}>
              Slots will populate as you describe your device needs.
            </div>
          )}
        </div>
      </div>

      {/* Remaining Slots Hints */}
      {remainingSlots && remainingSlots.length > 0 && (
        <div className="req-section">
          <span className="req-section-title">Missing Information</span>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
            <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              Provide these details to get tailored recommendations:
            </p>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.35rem' }}>
              {remainingSlots.map((slot) => (
                <button
                  key={slot}
                  type="button"
                  className="card-action-btn"
                  onClick={() => onSelectSlotHint && onSelectSlotHint(slot)}
                  title={`Specify ${slot}`}
                >
                  <Sparkles size={11} color="var(--accent-primary)" />
                  <span>Specify {slot.replace(/_/g, ' ')}</span>
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Verified Hard Specs */}
      {Object.keys(hardConstraints).length > 0 && (
        <div className="req-section">
          <span className="req-section-title">Hard Constraints</span>
          <div className="specs-pills-row">
            {Object.entries(hardConstraints).map(([k, v]) => {
              if (v == null || v === '' || (Array.isArray(v) && v.length === 0)) return null;
              return (
                <span key={k} className="spec-pill">
                  <strong>{k.replace(/_/g, ' ')}:</strong>
                  {typeof v === 'boolean' ? (v ? 'Yes' : 'No') : Array.isArray(v) ? v.join(', ') : String(v)}
                </span>
              );
            })}
          </div>
        </div>
      )}
    </aside>
  );
}
