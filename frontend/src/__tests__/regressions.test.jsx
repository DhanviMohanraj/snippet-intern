import React from 'react';
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';

import RecommendationCard from '../components/RecommendationCard.jsx';
import MessageBubble from '../components/MessageBubble.jsx';

const laptop = {
  product_id: 'LAP-T1',
  name: 'Lumina UltraSlate 14',
  brand: 'Lumina',
  category: 'laptop',
  price_inr: 68000,
  rating: 4.3,
  overall_score: 93,
  verified_specs: { cpu: 'AMD Ryzen 5 7640U', ram_gb: 16, storage_gb: 512, weight_kg: 1.38 },
  reasoning: 'A strong fit for student and programming use.',
  advantages: [],
  trade_offs: [],
  limitations: [],
};

describe('Regression: error report 21 Oct 2026', () => {
  it('Error 25: spec label and value are separated by a space', () => {
    const { container } = render(<RecommendationCard recommendation={laptop} rank={1} />);
    const pills = [...container.querySelectorAll('.spec-pill')].map((el) => el.textContent);

    expect(pills).toContain('Processor: AMD Ryzen 5 7640U');
    expect(pills).toContain('RAM: 16 GB');
    expect(pills).toContain('Storage: 512 GB');
    expect(pills).toContain('Weight: 1.38 kg');
    // "Label:Value" with no space is what got entity-encoded (Processor:&#x41;MD).
    pills.forEach((p) => expect(p).not.toMatch(/:\S/));
  });

  it('Error 29: match score shown exactly as the 0-100 backend value', () => {
    render(<RecommendationCard recommendation={{ ...laptop, overall_score: 0.8 }} rank={1} />);
    // 0.8/100 rounds to 1%, it must not be inflated to 80%.
    expect(screen.queryByText('80% Match')).toBeNull();
  });

  it('Error 28: comparison summary and uncertainties render when provided', () => {
    const message = {
      role: 'assistant',
      content: 'Here are your options.',
      recommendations: [laptop],
      comparison_summary: 'Nexa CodeBook 14 is ₹18,001 cheaper.',
      remaining_uncertainties: ['Display resolution is not verified in the catalogue.'],
    };
    render(<MessageBubble message={message} />);
    expect(screen.getByText('Nexa CodeBook 14 is ₹18,001 cheaper.')).toBeTruthy();
    expect(screen.getByText('Display resolution is not verified in the catalogue.')).toBeTruthy();
  });
});

describe('Regression: phone spec labels (21 Oct 2026)', () => {
  it('formats camera and OS specs with proper labels', async () => {
    const { getSpecLabel, formatSpecValue } = await import('../utils/specFormatter.js');
    expect(getSpecLabel('camera_mp')).toBe('Camera');
    expect(formatSpecValue('camera_mp', 108)).toBe('108 MP');
    expect(getSpecLabel('os')).toBe('OS');
    expect(getSpecLabel('chipset')).toBe('Chipset');
  });
});
