/**
 * API client configuration and fetch wrapper.
 */

const BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
const USER_ID_STORAGE_KEY = 'electronics_agent_user_id';

/**
 * Retrieve or generate a persistent local user ID.
 */
export function getOrCreateUserId() {
  if (typeof window === 'undefined') return 'test-user-id';
  try {
    let uid = localStorage.getItem(USER_ID_STORAGE_KEY);
    if (!uid) {
      uid = 'user_' + Math.random().toString(36).substring(2, 10);
      localStorage.setItem(USER_ID_STORAGE_KEY, uid);
    }
    return uid;
  } catch {
    return 'user_default';
  }
}

export class ApiError extends Error {
  constructor(message, status = 0, details = null) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.details = details;
  }
}

/**
 * Universal fetch wrapper for the backend API.
 */
export async function apiRequest(endpoint, options = {}) {
  const url = `${BASE_URL.replace(/\/+$/, '')}${endpoint.startsWith('/') ? endpoint : `/${endpoint}`}`;
  const userId = options.userId || getOrCreateUserId();

  const headers = {
    'Content-Type': 'application/json',
    'X-User-ID': userId,
    ...(options.headers || {}),
  };

  const config = {
    ...options,
    headers,
  };

  try {
    const response = await fetch(url, config);

    let data = null;
    const contentType = response.headers.get('content-type') || '';
    if (contentType.includes('application/json')) {
      data = await response.json();
    } else {
      data = await response.text();
    }

    if (!response.ok) {
      let errorMsg = `Request failed with status ${response.status}`;
      if (data && typeof data === 'object') {
        if (data.detail) {
          errorMsg = typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail);
        } else if (data.message) {
          errorMsg = data.message;
        }
      }
      throw new ApiError(errorMsg, response.status, data);
    }

    return data;
  } catch (err) {
    if (err instanceof ApiError) {
      throw err;
    }
    // Network or parse errors
    throw new ApiError(
      err.message === 'Failed to fetch'
        ? 'Unable to connect to the backend server. Please verify the backend is running at ' + BASE_URL
        : err.message || 'Unknown network error',
      0,
      err
    );
  }
}
