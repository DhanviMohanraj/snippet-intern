import React, { useState, useEffect } from 'react';
import { Search, Star, MessageSquarePlus, Filter, Loader2 } from 'lucide-react';
import { getProducts, getCategories } from '../api/products.js';

export default function ProductBrowser({ onConsultAboutProduct }) {
  const [products, setProducts] = useState([]);
  const [categories, setCategories] = useState([]);
  const [selectedCategory, setSelectedCategory] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [maxPrice, setMaxPrice] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [totalProducts, setTotalProducts] = useState(0);

  useEffect(() => {
    loadCategories();
    fetchProducts();
  }, []);

  const loadCategories = async () => {
    try {
      const data = await getCategories();
      setCategories(Array.isArray(data) ? data : []);
    } catch (err) {
      console.warn('Failed to fetch categories:', err);
    }
  };

  const fetchProducts = async (cat = selectedCategory, maxP = maxPrice) => {
    setIsLoading(true);
    try {
      const params = {
        category: cat || undefined,
        max_price: maxP ? Number(maxP) : undefined,
        page_size: 24,
      };
      const data = await getProducts(params);
      if (data && Array.isArray(data.items)) {
        setProducts(data.items);
        setTotalProducts(data.total || data.items.length);
      } else if (Array.isArray(data)) {
        setProducts(data);
        setTotalProducts(data.length);
      }
    } catch (err) {
      console.error('Failed to load products:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleCategorySelect = (cat) => {
    const newCat = selectedCategory === cat ? '' : cat;
    setSelectedCategory(newCat);
    fetchProducts(newCat, maxPrice);
  };

  const handlePriceFilter = (e) => {
    e.preventDefault();
    fetchProducts(selectedCategory, maxPrice);
  };

  const filteredProducts = products.filter((p) => {
    if (!searchQuery) return true;
    const q = searchQuery.toLowerCase();
    const nameMatch = (p.name || '').toLowerCase().includes(q);
    const brandMatch = (p.brand || '').toLowerCase().includes(q);
    return nameMatch || brandMatch;
  });

  const formatPrice = (val) => {
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR',
      maximumFractionDigits: 0,
    }).format(val || 0);
  };

  return (
    <div className="product-browser-view" role="region" aria-label="Product Catalogue">
      <div className="browser-filters-bar">
        <div className="filter-categories">
          <button
            type="button"
            className={`category-filter-chip ${selectedCategory === '' ? 'active' : ''}`}
            onClick={() => handleCategorySelect('')}
          >
            All Categories
          </button>
          {categories.map((c) => {
            const catName = typeof c === 'string' ? c : c.category;
            return (
              <button
                key={catName}
                type="button"
                className={`category-filter-chip ${selectedCategory === catName ? 'active' : ''}`}
                onClick={() => handleCategorySelect(catName)}
              >
                {catName}
              </button>
            );
          })}
        </div>

        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
          <div style={{ position: 'relative' }}>
            <input
              type="text"
              placeholder="Search catalogue..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="composer-textarea"
              style={{
                background: 'var(--bg-surface-elevated)',
                border: '1px solid var(--border-medium)',
                borderRadius: 'var(--radius-md)',
                padding: '0.4rem 0.75rem 0.4rem 2rem',
                fontSize: '0.82rem',
                width: '180px',
              }}
            />
            <Search size={14} style={{ position: 'absolute', left: '0.65rem', top: '0.6rem', color: 'var(--text-muted)' }} />
          </div>

          <form onSubmit={handlePriceFilter} style={{ display: 'flex', gap: '0.35rem' }}>
            <input
              type="number"
              placeholder="Max ₹"
              value={maxPrice}
              onChange={(e) => setMaxPrice(e.target.value)}
              className="composer-textarea"
              style={{
                background: 'var(--bg-surface-elevated)',
                border: '1px solid var(--border-medium)',
                borderRadius: 'var(--radius-md)',
                padding: '0.4rem 0.6rem',
                fontSize: '0.82rem',
                width: '100px',
              }}
            />
            <button type="submit" className="card-action-btn" title="Apply filter">
              <Filter size={13} />
            </button>
          </form>
        </div>
      </div>

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
        <span>Showing {filteredProducts.length} of {totalProducts} verified products in repository</span>
      </div>

      {isLoading ? (
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', flex: 1 }}>
          <Loader2 size={32} className="spinner" style={{ color: 'var(--accent-primary)' }} />
        </div>
      ) : (
        <div className="products-grid">
          {filteredProducts.map((p) => (
            <div key={p.id || p.product_id} className="catalogue-card">
              <div>
                <span className="product-brand-category">
                  {p.brand} • {p.category}
                </span>
                <h4 className="product-name" style={{ marginTop: '0.25rem', fontSize: '1rem' }}>
                  {p.name}
                </h4>

                <div className="price-rating-row" style={{ marginTop: '0.5rem' }}>
                  <span className="product-price" style={{ fontSize: '1.15rem' }}>
                    {formatPrice(p.price_inr)}
                  </span>
                  {p.rating && (
                    <div className="product-rating">
                      <Star size={13} fill="#fbbf24" stroke="#fbbf24" />
                      <span>{Number(p.rating).toFixed(1)}</span>
                    </div>
                  )}
                </div>

                {p.specs && (
                  <div className="specs-pills-row" style={{ marginTop: '0.65rem' }}>
                    {Object.entries(p.specs).slice(0, 4).map(([k, v]) => (
                      <span key={k} className="spec-pill" style={{ fontSize: '0.68rem' }}>
                        <strong>{k}:</strong> {String(v)}
                      </span>
                    ))}
                  </div>
                )}
              </div>

              <div className="card-action-row">
                <button
                  type="button"
                  className="card-action-btn"
                  onClick={() => onConsultAboutProduct && onConsultAboutProduct(p)}
                  title="Ask assistant about this product"
                  style={{ width: '100%', justifyContent: 'center' }}
                >
                  <MessageSquarePlus size={14} color="var(--accent-primary)" />
                  <span>Consult Assistant</span>
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
