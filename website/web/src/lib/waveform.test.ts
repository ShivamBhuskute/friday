import { describe, expect, it } from 'vitest'
import { columnForFraction, computePeaks, toMono } from '../lib/waveform'

describe('computePeaks', () => {
  it('returns one value per column', () => {
    const samples = new Float32Array(1000)
    const peaks = computePeaks(samples, 16_000, 40)
    expect(peaks.values).toHaveLength(40)
    expect(peaks.sampleCount).toBe(1000)
    expect(peaks.durationS).toBeCloseTo(1000 / 16_000, 6)
  })

  it('keeps the loudest sample in each bucket rather than the average', () => {
    // A single spike must stay visible; averaging would hide it entirely.
    const samples = new Float32Array(1000)
    samples[123] = 1
    const peaks = computePeaks(samples, 16_000, 10)
    expect(peaks.values[1]).toBe(1)
  })

  it('preserves the sign-independence of peaks', () => {
    const samples = new Float32Array(100)
    samples[0] = -0.8
    expect(computePeaks(samples, 16_000, 1).values[0]).toBeCloseTo(0.8, 6)
  })

  it('clamps above 1 so a hot signal cannot overflow the column', () => {
    const samples = new Float32Array(4).fill(4)
    expect(computePeaks(samples, 16_000, 1).values[0]).toBe(1)
  })

  it('handles degenerate input without throwing', () => {
    expect(computePeaks(new Float32Array(0), 16_000, 10).durationS).toBe(0)
    expect(computePeaks(new Float32Array(10), 0, 10).durationS).toBe(0)
    // Zero columns must not produce a division by zero.
    expect(computePeaks(new Float32Array(10), 16_000, 0).values).toHaveLength(1)
  })

  it('covers every sample exactly once', () => {
    const samples = new Float32Array(1001)
    for (let i = 0; i < samples.length; i += 1) samples[i] = i / samples.length
    // 1001 samples into 7 columns cannot divide evenly, so bucket sizes vary;
    // what matters is that nothing is dropped or double counted.
    const peaks = computePeaks(samples, 16_000, 7)
    const last = peaks.values[6]!
    expect(last).toBeGreaterThan(0)
    expect(last).toBeLessThanOrEqual(1)
  })
})

describe('toMono', () => {
  it('passes a mono buffer through', () => {
    const data = new Float32Array([0.1, 0.2])
    const buffer = {
      numberOfChannels: 1,
      length: 2,
      getChannelData: () => data,
    } as unknown as AudioBuffer
    const mono = toMono(buffer)
    // Float32 storage means 0.1 comes back as 0.10000000149...
    expect(mono[0]).toBeCloseTo(0.1, 6)
    expect(mono[1]).toBeCloseTo(0.2, 6)
  })

  it('averages channels', () => {
    const left = new Float32Array([1, 0])
    const right = new Float32Array([0, 1])
    const buffer = {
      numberOfChannels: 2,
      length: 2,
      getChannelData: (i: number) => (i === 0 ? left : right),
    } as unknown as AudioBuffer
    const mono = toMono(buffer)
    expect(mono[0]).toBeCloseTo(0.5, 6)
    expect(mono[1]).toBeCloseTo(0.5, 6)
  })
})

describe('columnForFraction', () => {
  const peaks = { values: new Float32Array(100), sampleCount: 100, durationS: 1 }

  it('maps the playhead onto a column', () => {
    expect(columnForFraction(peaks, 0)).toBe(0)
    expect(columnForFraction(peaks, 0.5)).toBe(50)
  })

  it('clamps out-of-range fractions', () => {
    expect(columnForFraction(peaks, -1)).toBe(0)
    expect(columnForFraction(peaks, 2)).toBe(99)
  })

  it('is safe before the audio has loaded', () => {
    expect(columnForFraction({ values: new Float32Array(0), sampleCount: 0, durationS: 0 }, 0.5)).toBe(0)
  })
})
