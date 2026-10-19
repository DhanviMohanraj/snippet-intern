import { describe, it, expect, vi, beforeEach } from 'vitest';
import { apiRequest, ApiError, getOrCreateUserId } from '../api/client.js';
import {
  createConversation,
  sendMessage,
  getConversation,
  getMessages,
  deleteConversation,
} from '../api/conversations.js';
import { getProducts, getCategories, getHealth } from '../api/products.js';

describe('API Client Layer', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    localStorage.clear();
  });

  it('persists and retrieves user ID in localStorage', () => {
    const id1 = getOrCreateUserId();
    expect(id1).toMatch(/^user_/);
    const id2 = getOrCreateUserId();
    expect(id2).toBe(id1);
  });

  it('apiRequest attaches X-User-ID and parses JSON response on 200', async () => {
    const mockResponse = { id: 'conv-123', state: 'active' };
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      headers: {
        get: (h) => (h === 'content-type' ? 'application/json' : null),
      },
      json: async () => mockResponse,
    });

    const res = await apiRequest('/api/v1/conversations', {
      method: 'POST',
      userId: 'test-user',
    });

    expect(res).toEqual(mockResponse);
    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining('/api/v1/conversations'),
      expect.objectContaining({
        headers: expect.objectContaining({
          'Content-Type': 'application/json',
          'X-User-ID': 'test-user',
        }),
      })
    );
  });

  it('apiRequest throws structured ApiError on 404 or 500', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 404,
      headers: {
        get: () => 'application/json',
      },
      json: async () => ({ detail: 'Conversation not found' }),
    });

    await expect(apiRequest('/api/v1/conversations/unknown-id')).rejects.toThrow(
      'Conversation not found'
    );
  });

  it('conversations API methods invoke proper endpoints', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      headers: { get: () => 'application/json' },
      json: async () => ({ ok: true }),
    });

    await createConversation('user-1', { tag: 'test' });
    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining('/api/v1/conversations'),
      expect.objectContaining({ method: 'POST' })
    );

    await sendMessage('conv-1', 'Need a laptop', 'user-1');
    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining('/api/v1/conversations/conv-1/messages'),
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ message: 'Need a laptop' }),
      })
    );

    await getConversation('conv-1');
    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining('/api/v1/conversations/conv-1'),
      expect.objectContaining({ method: 'GET' })
    );

    await getMessages('conv-1');
    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining('/api/v1/conversations/conv-1/messages'),
      expect.objectContaining({ method: 'GET' })
    );

    await deleteConversation('conv-1');
    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining('/api/v1/conversations/conv-1'),
      expect.objectContaining({ method: 'DELETE' })
    );
  });

  it('products API serializes query parameters correctly', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      headers: { get: () => 'application/json' },
      json: async () => ({ items: [], total: 0 }),
    });

    await getProducts({ category: 'laptop', max_price: 70000, page_size: 10 });
    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringMatching(/category=laptop.*max_price=70000.*page_size=10/),
      expect.objectContaining({ method: 'GET' })
    );

    await getCategories();
    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining('/api/v1/products/categories'),
      expect.objectContaining({ method: 'GET' })
    );

    await getHealth();
    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining('/api/v1/health'),
      expect.objectContaining({ method: 'GET' })
    );
  });
});
