import React, { useState, useEffect, useCallback } from 'react';
import Header from './components/Header.jsx';
import Sidebar from './components/Sidebar.jsx';
import ChatArea from './components/ChatArea.jsx';
import MessageComposer from './components/MessageComposer.jsx';
import RequirementsPanel from './components/RequirementsPanel.jsx';
import ProductBrowser from './components/ProductBrowser.jsx';

import {
  createConversation,
  sendMessage,
  getMessages,
  deleteConversation,
} from './api/conversations.js';
import { getHealth } from './api/products.js';
import { getOrCreateUserId } from './api/client.js';

const SESSIONS_STORAGE_KEY = 'specsage_conversations_v1';

export default function App() {
  const [userId] = useState(() => getOrCreateUserId());
  const [activeTab, setActiveTab] = useState('chat');
  const [conversations, setConversations] = useState(() => {
    try {
      const saved = localStorage.getItem(SESSIONS_STORAGE_KEY);
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  });

  const [currentConversationId, setCurrentConversationId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [requirements, setRequirements] = useState(null);
  const [canRecommend, setCanRecommend] = useState(false);
  const [remainingSlots, setRemainingSlots] = useState([]);
  const [slotStatus, setSlotStatus] = useState({});
  const [contradictions, setContradictions] = useState([]);
  const [composerInitialText, setComposerInitialText] = useState('');

  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState(null);
  const [lastFailedMessage, setLastFailedMessage] = useState(null);

  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [requirementsOpen, setRequirementsOpen] = useState(true);
  const [healthStatus, setHealthStatus] = useState(null);

  // Persist sessions
  useEffect(() => {
    try {
      localStorage.setItem(SESSIONS_STORAGE_KEY, JSON.stringify(conversations));
    } catch (e) {
      console.warn('Failed to save sessions to localStorage:', e);
    }
  }, [conversations]);

  // Load backend health on mount
  const checkHealth = useCallback(async () => {
    try {
      const data = await getHealth();
      setHealthStatus(data);
    } catch (err) {
      setHealthStatus({ status: 'error', error: err.message });
    }
  }, []);

  useEffect(() => {
    checkHealth();
  }, [checkHealth]);

  // Initialize or restore conversation
  useEffect(() => {
    if (conversations.length > 0 && !currentConversationId) {
      handleSelectConversation(conversations[0].id);
    }
  }, []);

  const handleNewConversation = async () => {
    setIsLoading(true);
    setErrorMessage(null);
    try {
      const resp = await createConversation(userId, { client: 'web-ui' });
      const newConvId = resp.id || resp.conversation_id;
      const newConvItem = {
        id: newConvId,
        preview: 'New consultation',
        turn_count: 0,
        updated_at: new Date().toISOString(),
      };

      setConversations((prev) => [newConvItem, ...prev.filter((c) => c.id !== newConvId)]);
      setCurrentConversationId(newConvId);
      setMessages([]);
      setRequirements(null);
      setCanRecommend(false);
      setRemainingSlots([]);
      setSlotStatus({});
      setContradictions([]);
      setComposerInitialText('');
      setActiveTab('chat');
    } catch (err) {
      setErrorMessage(err.message || 'Failed to initialize a new conversation.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleSelectConversation = async (convId) => {
    if (!convId || convId === currentConversationId) return;
    setIsLoading(true);
    setErrorMessage(null);
    setCurrentConversationId(convId);

    try {
      const history = await getMessages(convId);
      if (history && Array.isArray(history.messages)) {
        setMessages(history.messages);
        // Find latest assistant turn to restore requirements state
        for (let i = history.messages.length - 1; i >= 0; i--) {
          const m = history.messages[i];
          if (m.role === 'assistant' && m.metadata) {
            if (m.metadata.requirements) setRequirements(m.metadata.requirements);
            if (m.metadata.can_recommend != null) setCanRecommend(m.metadata.can_recommend);
            if (m.metadata.remaining_slots) setRemainingSlots(m.metadata.remaining_slots);
            if (m.metadata.slot_status) setSlotStatus(m.metadata.slot_status);
            if (m.metadata.unresolved_contradictions) setContradictions(m.metadata.unresolved_contradictions);
            break;
          }
        }
      }
    } catch (err) {
      console.warn('Could not load messages history:', err);
      // Keep whatever is currently held or start fresh
    } finally {
      setIsLoading(false);
    }
  };

  const handleDeleteConversation = async (convId) => {
    try {
      await deleteConversation(convId);
    } catch (e) {
      console.warn('Backend delete returned warning:', e);
    }

    const updated = conversations.filter((c) => c.id !== convId);
    setConversations(updated);

    if (currentConversationId === convId) {
      if (updated.length > 0) {
        handleSelectConversation(updated[0].id);
      } else {
        setCurrentConversationId(null);
        setMessages([]);
        setRequirements(null);
        setCanRecommend(false);
        setRemainingSlots([]);
        setSlotStatus({});
        setContradictions([]);
      }
    }
  };

  const handleSendMessage = async (text) => {
    if (!text || !text.trim()) return;
    const cleanText = text.trim();
    setErrorMessage(null);
    setLastFailedMessage(null);

    let activeId = currentConversationId;

    // Create a new session if none exists
    if (!activeId) {
      setIsLoading(true);
      try {
        const resp = await createConversation(userId, { client: 'web-ui' });
        activeId = resp.id || resp.conversation_id;
        setCurrentConversationId(activeId);
        const newSession = {
          id: activeId,
          preview: cleanText.substring(0, 32) + (cleanText.length > 32 ? '...' : ''),
          turn_count: 0,
          updated_at: new Date().toISOString(),
        };
        setConversations((prev) => [newSession, ...prev]);
      } catch (err) {
        setIsLoading(false);
        setErrorMessage(err.message || 'Failed to start conversation.');
        setLastFailedMessage(cleanText);
        return;
      }
    }

    // Add user turn immediately to the chat stream
    const userMessage = {
      role: 'user',
      content: cleanText,
      turn_index: (messages.length / 2) + 1,
      created_at: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);

    try {
      const turnResp = await sendMessage(activeId, cleanText, userId);

      const assistantMessage = {
        role: 'assistant',
        content: turnResp.message || '',
        turn_index: turnResp.turn_count,
        action: turnResp.action,
        requires_clarification: turnResp.requires_clarification,
        clarification_question: turnResp.clarification_question,
        unresolved_contradictions: turnResp.unresolved_contradictions || [],
        recommendations: turnResp.recommendations || [],
        can_recommend: turnResp.can_recommend,
        remaining_slots: turnResp.remaining_slots || [],
        slot_status: turnResp.slot_status || {},
        created_at: new Date().toISOString(),
        metadata: {
          requirements: turnResp.requirements,
          can_recommend: turnResp.can_recommend,
          remaining_slots: turnResp.remaining_slots,
          slot_status: turnResp.slot_status,
          unresolved_contradictions: turnResp.unresolved_contradictions,
        },
      };

      setMessages((prev) => [...prev, assistantMessage]);

      // Update state panels
      if (turnResp.requirements) setRequirements(turnResp.requirements);
      if (turnResp.can_recommend != null) setCanRecommend(turnResp.can_recommend);
      if (turnResp.remaining_slots) setRemainingSlots(turnResp.remaining_slots);
      if (turnResp.slot_status) setSlotStatus(turnResp.slot_status);
      if (turnResp.unresolved_contradictions) setContradictions(turnResp.unresolved_contradictions);

      // Update conversation metadata
      setConversations((prev) =>
        prev.map((c) =>
          c.id === activeId
            ? {
                ...c,
                preview: c.preview === 'New consultation' ? cleanText.substring(0, 32) + '...' : c.preview,
                turn_count: turnResp.turn_count,
                updated_at: new Date().toISOString(),
              }
            : c
        )
      );
    } catch (err) {
      setErrorMessage(err.message || 'Failed to get recommendation from agent.');
      setLastFailedMessage(cleanText);
    } finally {
      setIsLoading(false);
    }
  };

  const handleRetry = () => {
    if (lastFailedMessage) {
      handleSendMessage(lastFailedMessage);
    }
  };

  const handleAskFollowUp = (productName) => {
    setActiveTab('chat');
    setComposerInitialText(`Can you tell me more about the advantages and trade-offs of the ${productName}?`);
  };

  const handleSelectStarterPrompt = (promptText) => {
    handleSendMessage(promptText);
  };

  const handleSelectSlotHint = (slotName) => {
    const hint = `Regarding the ${slotName.replace(/_/g, ' ')}, I need `;
    setComposerInitialText(hint);
  };

  const handleConsultAboutProduct = (product) => {
    setActiveTab('chat');
    handleSendMessage(`I'm looking at the ${product.brand} ${product.name} (₹${product.price_inr}). Is this suitable for my requirements?`);
  };

  return (
    <div className="app-container">
      <Header
        activeTab={activeTab}
        onTabChange={setActiveTab}
        onNewConsultation={handleNewConversation}
        healthStatus={healthStatus}
        onRefreshHealth={checkHealth}
        sidebarOpen={sidebarOpen}
        onToggleSidebar={() => setSidebarOpen((prev) => !prev)}
        requirementsOpen={requirementsOpen}
        onToggleRequirements={() => setRequirementsOpen((prev) => !prev)}
      />

      <div className="main-layout">
        <Sidebar
          conversations={conversations}
          activeConversationId={currentConversationId}
          onSelectConversation={handleSelectConversation}
          onNewConversation={handleNewConversation}
          onDeleteConversation={handleDeleteConversation}
          userId={userId}
          isOpen={sidebarOpen}
        />

        {activeTab === 'chat' ? (
          <main className="chat-workspace">
            <div className="chat-content-container">
              <ChatArea
                messages={messages}
                isLoading={isLoading}
                onSelectStarterPrompt={handleSelectStarterPrompt}
                onAskFollowUp={handleAskFollowUp}
              />

              <RequirementsPanel
                requirements={requirements}
                canRecommend={canRecommend}
                remainingSlots={remainingSlots}
                slotStatus={slotStatus}
                contradictions={contradictions}
                onSelectSlotHint={handleSelectSlotHint}
                isOpen={requirementsOpen}
              />
            </div>

            <MessageComposer
              onSendMessage={handleSendMessage}
              isLoading={isLoading}
              errorMessage={errorMessage}
              onRetry={lastFailedMessage ? handleRetry : null}
              initialText={composerInitialText}
            />
          </main>
        ) : (
          <ProductBrowser onConsultAboutProduct={handleConsultAboutProduct} />
        )}
      </div>
    </div>
  );
}
