import type { Turn } from '../lib/types'

/** A turn with sensible defaults, so a test only states what it cares about. */
export function makeTurn(overrides: Partial<Turn> = {}): Turn {
  return {
    id: 'turn-0001',
    seq: 1,
    created_at: '2026-09-26T07:28:03.000Z',
    updated_at: '2026-09-26T07:28:04.000Z',
    state: 'done',
    audio_url: null,
    duration_s: null,
    sample_rate: null,
    channels: null,
    source: 'esp32',
    transcript: 'what is the weather in Pune right now',
    confidence: 0.94,
    answer: 'It is 28 degrees Celsius in Pune.',
    error: null,
    tool_calls: [],
    transcript_ms: 104,
    llm_ms: 2210,
    ...overrides,
  }
}

export function makeToolCall(overrides: Partial<Turn['tool_calls'][number]> = {}) {
  return {
    name: 'get_weather',
    arguments: { city: 'Pune' },
    result: { result: '28 C, clear' },
    error: null,
    duration_ms: 1800,
    ...overrides,
  }
}
