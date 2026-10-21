import React from 'react';
import { Star, CheckCircle, AlertCircle, MessageCircleQuestion, Sparkles, AlertTriangle } from 'lucide-react';
import { getSpecLabel, formatSpecValue } from '../utils/specFormatter.js';

export default function RecommendationCard({ recommendation, rank, onAskFollowUp }) {
  if (!recommendation) return null;

  const {
    product_id,
    name,
    brand,
    category,
    price_inr,
    rating,
    verified_specs = {},
    overall_score = 0,
    reasoning = '',
    advantages = [],
    trade_offs = [],
    limitations = [],
  } = recommendation;

  const formatPrice = (val) => {
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR',
      maximumFractionDigits: 0,
    }).format(val || 0);
  };

  // Backend overall_score is always on a 0-100 scale; display it as-is so the
  // card never disagrees with the API (a score below 1 must not be multiplied).
  const scorePercentage = Math.round(Math.max(0, Math.min(100, Number(overall_score) || 0)));

  const specsList = Object.entries(verified_specs).filter(
    ([k, v]) => v != null && v !== '' && !(Array.isArray(v) && v.length === 0)
  );

  return (
    <article className="recommendation-card" data-testid={`product-rec-${product_id}`}>
      <div className="card-top-row">
        <div className="product-brand-title">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
            {rank != null && (
              <span
                className="product-rank-badge"
                style={{
                  background: 'var(--accent-primary)',
                  color: '#ffffff',
                  fontSize: '0.72rem',
                  fontWeight: 800,
                  padding: '0.15rem 0.45rem',
                  borderRadius: 'var(--radius-sm)',
                }}
              >
                #{rank}
              </span>
            )}
            <span className="product-brand-category">
              {brand} • {category}
            </span>
          </div>
          <h4 className="product-name">{name}</h4>
        </div>

        {scorePercentage > 0 && (
          <div className="match-score-pill" title={`Overall match score: ${scorePercentage}%`}>
            <Sparkles size={12} />
            <span>{scorePercentage}% Match</span>
          </div>
        )}
      </div>

      <div className="price-rating-row">
        <span className="product-price">{formatPrice(price_inr)}</span>
        {rating > 0 && (
          <div className="product-rating" title={`Rating: ${rating} / 5`}>
            <Star size={14} fill="#fbbf24" stroke="#fbbf24" />
            <span>{rating.toFixed(1)}</span>
          </div>
        )}
      </div>

      {specsList.length > 0 && (
        <div className="specs-pills-row">
          {specsList.map(([key, val]) => (
            <span key={key} className="spec-pill">
              {/* Explicit space: JSX drops the line break between elements, which
                  rendered "Processor:AMD…" and got entity-encoded when copied. */}
              <strong>{getSpecLabel(key)}:</strong>{' '}
              {formatSpecValue(key, val)}
            </span>
          ))}
        </div>
      )}

      {reasoning && (
        <div className="recommendation-reasoning">
          <p>{reasoning}</p>
        </div>
      )}

      {(advantages.length > 0 || trade_offs.length > 0) && (
        <div className="pros-cons-grid">
          {advantages.length > 0 && (
            <div className="pros-column">
              <span className="pros-heading">
                <CheckCircle size={12} /> Advantages
              </span>
              {advantages.map((adv, i) => (
                <div key={i} className="pro-item">
                  <span>•</span>
                  <span>{adv}</span>
                </div>
              ))}
            </div>
          )}

          {trade_offs.length > 0 && (
            <div className="cons-column">
              <span className="cons-heading">
                <AlertCircle size={12} /> Trade-offs
              </span>
              {trade_offs.map((con, i) => (
                <div key={i} className="con-item">
                  <span>•</span>
                  <span>{con}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {limitations.length > 0 && (
        <div
          className="limitations-callout"
          style={{
            background: 'rgba(234, 179, 8, 0.08)',
            border: '1px solid rgba(234, 179, 8, 0.25)',
            borderRadius: 'var(--radius-md)',
            padding: '0.65rem 0.85rem',
            marginTop: '0.4rem',
          }}
        >
          <span
            style={{
              fontSize: '0.74rem',
              fontWeight: 700,
              textTransform: 'uppercase',
              color: '#facc15',
              display: 'flex',
              alignItems: 'center',
              gap: '0.35rem',
              marginBottom: '0.3rem',
            }}
          >
            <AlertTriangle size={13} /> Unverified / Missing Specifications
          </span>
          {limitations.map((lim, i) => (
            <div
              key={i}
              style={{
                fontSize: '0.76rem',
                color: 'var(--text-secondary)',
                lineHeight: 1.35,
                display: 'flex',
                gap: '0.35rem',
              }}
            >
              <span>•</span>
              <span>{lim}</span>
            </div>
          ))}
        </div>
      )}

      {onAskFollowUp && (
        <div className="card-action-row">
          <button
            type="button"
            className="card-action-btn"
            onClick={() => onAskFollowUp(name)}
            title={`Ask more details about ${name}`}
          >
            <MessageCircleQuestion size={13} />
            <span>Ask about this model</span>
          </button>
        </div>
      )}
    </article>
  );
}

