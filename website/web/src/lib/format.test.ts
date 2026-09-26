import { describe, expect, it } from 'vitest'
import {
  formatAgo,
  formatBytes,
  formatClock,
  formatConfidence,
  formatDuration,
  formatMs,
  summariseResult,
} from '../lib/format'

describe('formatDuration', () => {
  it('renders sub-second clips in milliseconds', () => {
    expect(formatDuration(0.32)).toBe('320ms')
    expect(formatDuration(0.999)).toBe('999ms')
  })

  it('gives short clips two decimals and long clips one', () => {
    expect(formatDuration(1.44)).toBe('1.44s')
    expect(formatDuration(42.16)).toBe('42.2s')
  })

  it('switches to minutes', () => {
    expect(formatDuration(75)).toBe('1m 15s')
  })

  it('treats missing values as unknown', () => {
    expect(formatDuration(null)).toBe('--')
    expect(formatDuration(undefined)).toBe('--')
    expect(formatDuration(Number.NaN)).toBe('--')
  })
})

describe('formatMs', () => {
  it('never renders a bare zero', () => {
    expect(formatMs(0)).toBe('<1ms')
  })

  it('switches to seconds past a second', () => {
    expect(formatMs(340)).toBe('340ms')
    expect(formatMs(1500)).toBe('1.50s')
  })

  it('treats missing values as unknown', () => {
    expect(formatMs(null)).toBe('--')
  })
})

describe('formatClock', () => {
  it('renders a wall-clock time', () => {
    const shown = formatClock('2026-09-26T07:28:03Z')
    expect(shown).toMatch(/^\d{2}:\d{2}:\d{2}$/)
  })

  it('rejects junk instead of printing "Invalid Date"', () => {
    expect(formatClock('not a date')).toBe('--')
    expect(formatClock(null)).toBe('--')
  })
})

describe('formatConfidence', () => {
  it('renders a whole percentage', () => {
    expect(formatConfidence(0.874)).toBe('87%')
  })

  it('clamps out-of-range values rather than printing 140%', () => {
    expect(formatConfidence(1.4)).toBe('100%')
    expect(formatConfidence(-0.2)).toBe('0%')
  })

  it('is null when the model gave no score', () => {
    expect(formatConfidence(null)).toBeNull()
    expect(formatConfidence(undefined)).toBeNull()
  })
})

describe('formatBytes', () => {
  it('scales to the right unit', () => {
    expect(formatBytes(0)).toBe('0 B')
    expect(formatBytes(512)).toBe('512 B')
    expect(formatBytes(2048)).toBe('2.0 KB')
    expect(formatBytes(5 * 1024 * 1024)).toBe('5.0 MB')
  })
})

describe('formatAgo', () => {
  const now = Date.parse('2026-09-26T12:00:00Z')

  it('describes recent times', () => {
    expect(formatAgo('2026-09-26T11:59:58Z', now)).toBe('just now')
    expect(formatAgo('2026-09-26T11:59:30Z', now)).toBe('30s ago')
    expect(formatAgo('2026-09-26T11:55:00Z', now)).toBe('5m ago')
    expect(formatAgo('2026-09-26T09:00:00Z', now)).toBe('3h ago')
  })

  it('never goes negative for clock skew', () => {
    expect(formatAgo('2026-09-26T12:05:00Z', now)).toBe('just now')
  })
})

describe('summariseResult', () => {
  it('prefers the field a human wants to read', () => {
    expect(summariseResult({ result: '161' })).toBe('161')
    expect(summariseResult({ city: 'Pune', temperature_c: 28 })).toBe('{city, temperature_c}')
  })

  it('passes scalars through', () => {
    expect(summariseResult('plain')).toBe('plain')
    expect(summariseResult(42)).toBe('42')
    expect(summariseResult(true)).toBe('true')
    expect(summariseResult(null)).toBe('no result')
  })

  it('does not print an unbounded object', () => {
    const wide = Object.fromEntries(Array.from({ length: 40 }, (_, i) => [`k${i}`, i]))
    const summary = summariseResult(wide)
    expect(summary.length).toBeLessThan(40)
  })
})
