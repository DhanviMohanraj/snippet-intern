import React from 'react';
import { Star, CheckCircle, AlertCircle, MessageCircleQuestion, Sparkles } from 'lucide-react';

export default function RecommendationCard({ recommendation, onAskFollowUp }) {
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
  } = recommendation;

  const formatPrice = (val) => {
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR',
      maximumFractionDigits: 0,
    }).format(val || 0);
  };

  const scorePercentage = Math.round((overall_score > 1 ? overall_score : overall_score * 100));

  const specsList = Object.entries(verified_specs).filter(
    ([k, v]) => v != null && v !== '' && !(Array.isArray(v) && v.length === 0)
  );

  return (
    <article className="recommendation-card" data-testid={`product-rec-${product_id}`}>
      <div className="card-top-row">
        <div className="product-brand-title">
          <span className="product-brand-category">
            {brand} • {category}
          </span>
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
              <strong>{key.replace(/_/g, ' ')}:</strong>
              {Array.isArray(val) ? val.join(', ') : String(val)}
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
