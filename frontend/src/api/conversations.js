/**
 * Conversation API endpoints integration.
 */
import { apiRequest } from './client.js';

export async function createConversation(userId = null, metadata = {}) {
  return apiRequest('/api/v1/conversations', {
    method: 'POST',
    userId,
    body: JSON.stringify({
      user_id: userId,
      metadata: metadata || {},
    }),
  });
}

export async function sendMessage(conversationId, message, userId = null) {
  return apiRequest(`/api/v1/conversations/${conversationId}/messages`, {
    method: 'POST',
    userId,
    body: JSON.stringify({
      message: message.trim(),
    }),
  });
}

export async function getConversation(conversationId) {
  return apiRequest(`/api/v1/conversations/${conversationId}`, {
    method: 'GET',
  });
}

export async function getMessages(conversationId) {
  return apiRequest(`/api/v1/conversations/${conversationId}/messages`, {
    method: 'GET',
  });
}

export async function deleteConversation(conversationId) {
  return apiRequest(`/api/v1/conversations/${conversationId}`, {
    method: 'DELETE',
  });
}
