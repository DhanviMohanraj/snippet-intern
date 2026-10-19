/**
 * Products and System Health API endpoints integration.
 */
import { apiRequest } from './client.js';

export async function getProducts(params = {}) {
  const query = new URLSearchParams();
  if (params.category) query.append('category', params.category);
  if (params.min_price != null) query.append('min_price', String(params.min_price));
  if (params.max_price != null) query.append('max_price', String(params.max_price));
  if (params.use_case) query.append('use_case', params.use_case);
  if (params.sort) query.append('sort', params.sort);
  if (params.page != null) query.append('page', String(params.page));
  if (params.page_size != null) query.append('page_size', String(params.page_size));

  const qs = query.toString();
  return apiRequest(`/api/v1/products${qs ? `?${qs}` : ''}`, {
    method: 'GET',
  });
}

export async function getProductById(productId) {
  return apiRequest(`/api/v1/products/${encodeURIComponent(productId)}`, {
    method: 'GET',
  });
}

export async function getCategories() {
  return apiRequest('/api/v1/products/categories', {
    method: 'GET',
  });
}

export async function getHealth() {
  return apiRequest('/api/v1/health', {
    method: 'GET',
  });
}
