/**
 * Peak extraction for the waveform.
 *
 * The canvas draws `peaks` (a max-absolute amplitude per column) rather than
 * raw samples: a 10-second clip is 160 000 samples, and drawing 160 000 lines
 * per repaint at 60fps is what makes hand-rolled visualisers stutter. Reducing
 * to one column per device pixel means the draw cost is bounded by the width of
 * the element, not the length of the recording.
 */

export interface Peaks {
  /** One value in [0, 1] per column. */
  values: Float32Array
  /** Number of samples summarised. */
  sampleCount: number
  durationS: number
}

/**
 * Reduce an AudioBuffer to per-column peaks.
 *
 * `columns` is how many buckets to produce. Each bucket keeps the loudest
 * absolute sample it saw, so a short spike stays visible instead of being
 * averaged away into invisibility.
 */
export function computePeaks(
  samples: Float32Array,
  sampleRate: number,
  columns: number,
): Peaks {
  const width = Math.max(1, Math.floor(columns))
  const values = new Float32Array(width)
  const total = samples.length

  if (total === 0 || sampleRate <= 0) {
    return { values, sampleCount: 0, durationS: 0 }
  }

  for (let column = 0; column < width; column += 1) {
    // Sample the bucket boundaries directly rather than accumulating with a
    // per-sample modulo, which is measurably slower on 160k samples.
    const start = Math.floor((column * total) / width)
    const end = Math.max(start + 1, Math.floor(((column + 1) * total) / width))
    let peak = 0
    for (let i = start; i < end && i < total; i += 1) {
      const magnitude = samples[i]! < 0 ? -samples[i]! : samples[i]!
      if (magnitude > peak) peak = magnitude
    }
    values[column] = peak > 1 ? 1 : peak
  }

  return { values, sampleCount: total, durationS: total / sampleRate }
}

/** Mixed-down mono samples, for a buffer that may have several channels. */
export function toMono(buffer: AudioBuffer): Float32Array {
  const channels = buffer.numberOfChannels
  const length = buffer.length
  if (channels === 1) return buffer.getChannelData(0).slice()

  const out = new Float32Array(length)
  for (let c = 0; c < channels; c += 1) {
    const data = buffer.getChannelData(c)
    for (let i = 0; i < length; i += 1) out[i]! += data[i]! / channels
  }
  return out
}

/** Index of the first column at or after `fraction` of the clip. */
export function columnForFraction(peaks: Peaks, fraction: number): number {
  if (peaks.values.length === 0) return 0
  const clamped = Math.max(0, Math.min(1, fraction))
  return Math.min(
    peaks.values.length - 1,
    Math.floor(clamped * peaks.values.length),
  )
}
