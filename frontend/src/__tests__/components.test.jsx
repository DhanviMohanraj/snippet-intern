import React from 'react';
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';

import Header from '../components/Header.jsx';
import Sidebar from '../components/Sidebar.jsx';
import RecommendationCard from '../components/RecommendationCard.jsx';
import RequirementsPanel from '../components/RequirementsPanel.jsx';
import MessageBubble from '../components/MessageBubble.jsx';
import MessageComposer from '../components/MessageComposer.jsx';

describe('UI Components Unit Tests', () => {
  it('Header renders branding and healthy system badge', () => {
    const healthData = {
      status: 'ok',
      database: { status: 'ok' },
      llm: { status: 'ok' },
    };

    render(
      <Header
        activeTab="chat"
        onTabChange={() => {}}
        onNewConsultation={() => {}}
        healthStatus={healthData}
        onRefreshHealth={() => {}}
        sidebarOpen={false}
        onToggleSidebar={() => {}}
        requirementsOpen={true}
        onToggleRequirements={() => {}}
      />
    );

    expect(screen.getByText('SpecSage')).toBeInTheDocument();
    expect(screen.getByText('AI Advisor')).toBeInTheDocument();
    expect(screen.getByText('System Online')).toBeInTheDocument();
  });

  it('Header displays Degraded when DB or LLM is down', () => {
    const healthData = {
      status: 'degraded',
      database: { status: 'ok' },
      llm: { status: 'error' },
    };

    render(
      <Header
        activeTab="chat"
        onTabChange={() => {}}
        onNewConsultation={() => {}}
        healthStatus={healthData}
        onRefreshHealth={() => {}}
        sidebarOpen={false}
        onToggleSidebar={() => {}}
        requirementsOpen={true}
        onToggleRequirements={() => {}}
      />
    );

    expect(screen.getByText('Degraded')).toBeInTheDocument();
  });

  it('Sidebar displays sessions and triggers conversation selection', () => {
    const mockSelect = vi.fn();
    const mockDelete = vi.fn();
    const convs = [
      { id: 'conv-1', preview: 'Gaming laptop search', turn_count: 2 },
      { id: 'conv-2', preview: 'Budget phone search', turn_count: 1 },
    ];

    render(
      <Sidebar
        conversations={convs}
        activeConversationId="conv-1"
        onSelectConversation={mockSelect}
        onNewConversation={() => {}}
        onDeleteConversation={mockDelete}
        userId="user-test-12345"
        isOpen={true}
      />
    );

    expect(screen.getByText('Gaming laptop search')).toBeInTheDocument();
    expect(screen.getByText('Budget phone search')).toBeInTheDocument();
    expect(screen.getByText(/user-test-12345/)).toBeInTheDocument();

    fireEvent.click(screen.getByText('Budget phone search'));
    expect(mockSelect).toHaveBeenCalledWith('conv-2');
  });

  it('RecommendationCard renders product details, verified specs, pros and cons', () => {
    const mockAsk = vi.fn();
    const recommendation = {
      product_id: 'prod-101',
      name: 'ASUS ROG Zephyrus G14',
      brand: 'ASUS',
      category: 'laptop',
      price_inr: 125000,
      rating: 4.7,
      overall_score: 95,
      verified_specs: {
        ram_gb: 16,
        storage_gb: 1000,
        gpu: 'RTX 4060',
      },
      reasoning: 'Superb match for gaming and heavy programming workflows.',
      advantages: ['Ultraportable form factor', 'OLED 120Hz display'],
      trade_offs: ['RAM soldered on some SKUs'],
    };

    render(
      <RecommendationCard
        recommendation={recommendation}
        onAskFollowUp={mockAsk}
      />
    );

    expect(screen.getByText('ASUS ROG Zephyrus G14')).toBeInTheDocument();
    expect(screen.getByText(/ASUS • laptop/i)).toBeInTheDocument();
    expect(screen.getByText('95% Match')).toBeInTheDocument();
    expect(screen.getByText('₹1,25,000')).toBeInTheDocument();
    expect(screen.getByText('4.7')).toBeInTheDocument();
    expect(screen.getByText(/Superb match for gaming/)).toBeInTheDocument();
    expect(screen.getByText('Ultraportable form factor')).toBeInTheDocument();
    expect(screen.getByText('RAM soldered on some SKUs')).toBeInTheDocument();

    const askBtn = screen.getByRole('button', { name: /Ask about this model/i });
    fireEvent.click(askBtn);
    expect(mockAsk).toHaveBeenCalledWith('ASUS ROG Zephyrus G14');
  });

  it('RequirementsPanel shows category, budget and slot status engine correctly', () => {
    const mockHint = vi.fn();
    const requirements = {
      category: 'laptop',
      budget_max_inr: 75000,
      use_cases: ['college coding', 'python'],
      hard: {
        ram_gb: 16,
      },
    };

    const slotStatus = {
      category: 'answered',
      ram: 'answered',
      storage: 'pending',
      screen_size: 'pending',
    };

    render(
      <RequirementsPanel
        requirements={requirements}
        canRecommend={false}
        remainingSlots={['storage', 'screen_size']}
        slotStatus={slotStatus}
        contradictions={[]}
        onSelectSlotHint={mockHint}
        isOpen={true}
      />
    );

    expect(screen.getByText('Gathering Specs')).toBeInTheDocument();
    expect(screen.getByText(/₹75,000/)).toBeInTheDocument();
    expect(screen.getByText('college coding')).toBeInTheDocument();
    expect(screen.getByText('python')).toBeInTheDocument();

    const storageBtn = screen.getByRole('button', { name: /Specify storage/i });
    fireEvent.click(storageBtn);
    expect(mockHint).toHaveBeenCalledWith('storage');
  });

  it('MessageBubble displays clarification alert when clarification is needed', () => {
    const assistantMsg = {
      role: 'assistant',
      content: 'Could you please clarify your budget?',
      turn_index: 1,
      requires_clarification: true,
      clarification_question: 'What is your maximum budget in INR?',
      unresolved_contradictions: [],
      recommendations: [],
    };

    render(<MessageBubble message={assistantMsg} />);

    expect(screen.getByText('Could you please clarify your budget?')).toBeInTheDocument();
    expect(screen.getByText('Clarification Needed')).toBeInTheDocument();
    expect(screen.getByText('What is your maximum budget in INR?')).toBeInTheDocument();
  });

  it('MessageComposer submits text and supports keyboard shortcut', () => {
    const mockSend = vi.fn();
    render(<MessageComposer onSendMessage={mockSend} isLoading={false} />);

    const input = screen.getByRole('textbox', { name: /Message prompt input/i });
    const sendBtn = screen.getByRole('button', { name: /Send message/i });

    // Submit with button
    fireEvent.change(input, { target: { value: 'Looking for a tablet' } });
    fireEvent.click(sendBtn);
    expect(mockSend).toHaveBeenCalledWith('Looking for a tablet');

    // Submit with Enter key
    fireEvent.change(input, { target: { value: 'Under ₹40,000' } });
    fireEvent.keyDown(input, { key: 'Enter', shiftKey: false });
    expect(mockSend).toHaveBeenCalledWith('Under ₹40,000');

    // Shift + Enter should NOT submit
    mockSend.mockClear();
    fireEvent.change(input, { target: { value: 'Multiline\nquery' } });
    fireEvent.keyDown(input, { key: 'Enter', shiftKey: true });
    expect(mockSend).not.toHaveBeenCalled();
  });

  it('MessageComposer renders error message and retry button when failed', () => {
    const mockRetry = vi.fn();
    render(
      <MessageComposer
        onSendMessage={() => {}}
        isLoading={false}
        errorMessage="Backend service timeout"
        onRetry={mockRetry}
      />
    );

    expect(screen.getByText('Backend service timeout')).toBeInTheDocument();
    const retryBtn = screen.getByRole('button', { name: /Retry/i });
    fireEvent.click(retryBtn);
    expect(mockRetry).toHaveBeenCalled();
  });
});
