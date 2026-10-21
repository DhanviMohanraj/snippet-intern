import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import App from '../App.jsx';

describe('App Integration & Conversational Shopping Flow', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    localStorage.clear();
  });

  it('renders initial hero empty state and starter prompt cards', async () => {
    global.fetch = vi.fn().mockImplementation((url) => {
      if (url.includes('/api/v1/health')) {
        return Promise.resolve({
          ok: true,
          status: 200,
          headers: { get: () => 'application/json' },
          json: async () => ({
            status: 'ok',
            database: { status: 'ok' },
            llm: { status: 'ok' },
          }),
        });
      }
      return Promise.resolve({
        ok: true,
        status: 200,
        headers: { get: () => 'application/json' },
        json: async () => ({}),
      });
    });

    render(<App />);

    await waitFor(() => {
      expect(screen.getByText('System Online')).toBeInTheDocument();
    });

    expect(screen.getByText('What device are you looking for?')).toBeInTheDocument();
    expect(screen.getByText('College Coding Laptop')).toBeInTheDocument();
    expect(screen.getByText('High-Performance Gaming')).toBeInTheDocument();
  });

  it('conducts multi-turn consultation: starts session, clarifies slot, receives recommendations', async () => {
    let callCount = 0;

    global.fetch = vi.fn().mockImplementation((url, opts) => {
      // Health check
      if (url.includes('/api/v1/health')) {
        return Promise.resolve({
          ok: true,
          status: 200,
          headers: { get: () => 'application/json' },
          json: async () => ({
            status: 'ok',
            database: { status: 'ok' },
            llm: { status: 'ok' },
          }),
        });
      }

      // Create conversation
      if (url.includes('/api/v1/conversations') && opts.method === 'POST' && !url.includes('/messages')) {
        return Promise.resolve({
          ok: true,
          status: 200,
          headers: { get: () => 'application/json' },
          json: async () => ({
            id: 'conv-session-101',
            turn_count: 0,
            state: 'active',
          }),
        });
      }

      // Send message
      if (url.includes('/messages') && opts.method === 'POST') {
        callCount++;
        if (callCount === 1) {
          // Turn 1: Backend asks for RAM / storage clarification
          return Promise.resolve({
            ok: true,
            status: 200,
            headers: { get: () => 'application/json' },
            json: async () => ({
              conversation_id: 'conv-session-101',
              turn_count: 1,
              message: 'Great! To ensure smooth Python and VS Code performance, how much RAM and storage do you need?',
              action: 'ASK',
              requires_clarification: true,
              clarification_question: 'How much RAM and SSD storage do you prefer?',
              unresolved_contradictions: [],
              recommendations: [],
              can_recommend: false,
              remaining_slots: ['ram', 'storage'],
              slot_status: {
                category: 'answered',
                budget: 'answered',
                ram: 'pending',
                storage: 'pending',
              },
              requirements: {
                category: 'laptop',
                budget_max_inr: 70000,
                use_cases: ['college programming', 'VS Code'],
              },
            }),
          });
        }

        // Turn 2: User provides RAM/storage, agent recommends products
        return Promise.resolve({
          ok: true,
          status: 200,
          headers: { get: () => 'application/json' },
          json: async () => ({
            conversation_id: 'conv-session-101',
            turn_count: 2,
            message: 'Based on your programming requirements and ₹70,000 budget, here are the top matching laptops with 16GB RAM and 512GB SSD.',
            action: 'RECOMMEND',
            requires_clarification: false,
            clarification_question: null,
            unresolved_contradictions: [],
            recommendations: [
              {
                product_id: 'lap-01',
                name: 'Lenovo IdeaPad Slim 5',
                brand: 'Lenovo',
                category: 'laptop',
                price_inr: 64990,
                rating: 4.5,
                overall_score: 94,
                verified_specs: {
                  ram_gb: 16,
                  storage_gb: 512,
                  cpu: 'Intel Core i5-13500H',
                },
                reasoning: 'Exceptional multicore CPU for compiling code and 16GB dual-channel RAM.',
                advantages: ['16GB RAM included', 'Comfortable keyboard'],
                trade_offs: ['Average speaker output'],
              },
            ],
            can_recommend: true,
            remaining_slots: [],
            slot_status: {
              category: 'answered',
              budget: 'answered',
              ram: 'answered',
              storage: 'answered',
            },
            requirements: {
              category: 'laptop',
              budget_max_inr: 70000,
              hard: { ram_gb: 16, storage_gb: 512 },
            },
          }),
        });
      }

      return Promise.resolve({
        ok: true,
        status: 200,
        headers: { get: () => 'application/json' },
        json: async () => ({}),
      });
    });

    render(<App />);

    // Click starter prompt
    const promptCard = screen.getByText('College Coding Laptop');
    fireEvent.click(promptCard);

    // Wait for Turn 1 response with clarification question
    await waitFor(() => {
      expect(
        screen.getByText(/how much RAM and storage do you need\?/i)
      ).toBeInTheDocument();
    });

    expect(screen.getByText('Clarification Needed')).toBeInTheDocument();
    expect(screen.getByText('How much RAM and SSD storage do you prefer?')).toBeInTheDocument();
    expect(screen.getByText('Gathering Specs')).toBeInTheDocument();

    // Now user answers clarification in Turn 2
    const input = screen.getByRole('textbox', { name: /Message prompt input/i });
    fireEvent.change(input, { target: { value: 'I want 16GB RAM and 512GB SSD' } });
    fireEvent.keyDown(input, { key: 'Enter', shiftKey: false });

    // Wait for Turn 2 recommendations to render
    await waitFor(() => {
      expect(screen.getByText('Lenovo IdeaPad Slim 5')).toBeInTheDocument();
    });

    expect(screen.getByText('94% Match')).toBeInTheDocument();
    expect(screen.getByText('₹64,990')).toBeInTheDocument();
    expect(screen.getByText('Exceptional multicore CPU for compiling code and 16GB dual-channel RAM.')).toBeInTheDocument();
    expect(screen.getByText('Ready to Recommend')).toBeInTheDocument();
  });

  it('switches between AI Consultation and Product Catalogue views', async () => {
    global.fetch = vi.fn().mockImplementation((url) => {
      if (url.includes('/api/v1/health')) {
        return Promise.resolve({
          ok: true,
          status: 200,
          headers: { get: () => 'application/json' },
          json: async () => ({ status: 'ok', database: { status: 'ok' }, llm: { status: 'ok' } }),
        });
      }
      if (url.includes('/categories')) {
        return Promise.resolve({
          ok: true,
          status: 200,
          headers: { get: () => 'application/json' },
          json: async () => [{ category: 'laptop', count: 15 }, { category: 'smartphone', count: 20 }],
        });
      }
      if (url.includes('/products')) {
        return Promise.resolve({
          ok: true,
          status: 200,
          headers: { get: () => 'application/json' },
          json: async () => ({
            items: [
              {
                id: 'p-1',
                name: 'Dell XPS 13',
                brand: 'Dell',
                category: 'laptop',
                price_inr: 110000,
                rating: 4.6,
                specs: { ram_gb: 16 },
              },
            ],
            total: 1,
          }),
        });
      }
      return Promise.resolve({
        ok: true,
        status: 200,
        headers: { get: () => 'application/json' },
        json: async () => ({}),
      });
    });

    render(<App />);

    const catalogueTab = screen.getByRole('button', { name: /Catalogue Browser Mode/i });
    fireEvent.click(catalogueTab);

    await waitFor(() => {
      expect(screen.getByText('Dell XPS 13')).toBeInTheDocument();
    });

    expect(screen.getByText(/Dell • laptop/i)).toBeInTheDocument();
    expect(screen.getByText('₹1,10,000')).toBeInTheDocument();
  });
});
