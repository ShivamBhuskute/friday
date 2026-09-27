import { useCallback, useEffect, useRef, useState } from 'react'
import { useAudioPlayer } from '../hooks/useAudioPlayer'
import { formatDuration } from '../lib/format'
import { columnForFraction, type Peaks } from '../lib/waveform'

/**
 * Waveform with a playhead.
 *
 * The peaks are drawn on a canvas rather than as DOM nodes: a 240-column
 * waveform is 240 elements, and every one of them would be a React child
 * re-created on each of the ~60 playhead updates per second. A canvas repaint
 * of the same data is effectively free.
 */

const BASE = '#2a3140'
const PLAYED = '#2dd4bf'
const CENTER = '#1e2330'

interface WaveformProps {
  url: string | null
  /** Fallback length, shown before the audio has been decoded. */
  durationS?: number | null
  disabled?: boolean
}

export function Waveform({ url, durationS, disabled = false }: WaveformProps) {
  const { playback, peaks, toggle, seek } = useAudioPlayer(url)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const [width, setWidth] = useState(0)
  const wrapRef = useRef<HTMLDivElement | null>(null)

  // Track the element width so the canvas gets a sensible backing-store size
  // and stays sharp on a high-DPI display.
  useEffect(() => {
    const element = wrapRef.current
    if (!element || typeof ResizeObserver === 'undefined') return
    const observer = new ResizeObserver((entries) => {
      const entry = entries[0]
      if (entry) setWidth(Math.floor(entry.contentRect.width))
    })
    observer.observe(element)
    setWidth(element.clientWidth)
    return () => observer.disconnect()
  }, [])

  const draw = useCallback(
    (data: Peaks | null, progress: number) => {
      const canvas = canvasRef.current
      if (!canvas || width <= 0) return
      const dpr = typeof devicePixelRatio === 'number' ? devicePixelRatio : 1
      const height = 56
      canvas.width = Math.floor(width * dpr)
      canvas.height = Math.floor(height * dpr)
      canvas.style.width = `${width}px`
      canvas.style.height = `${height}px`

      const ctx = canvas.getContext('2d')
      if (!ctx) return
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      ctx.clearRect(0, 0, width, height)

      // Baseline: without it a silent clip looks like an empty box.
      ctx.fillStyle = CENTER
      ctx.fillRect(0, height / 2 - 0.5, width, 1)

      if (!data) {
        // Skeleton bars while decoding, so the row does not jump when it lands.
        ctx.fillStyle = BASE
        for (let i = 0; i < 40; i += 1) {
          const barHeight = 6 + ((i * 7) % 5) * 3
          ctx.fillRect((i * width) / 40, (height - barHeight) / 2, 2, barHeight)
        }
        return
      }

      const values = data.values
      const playedTo = columnForFraction(data, progress)
      const columns = values.length
      const barWidth = Math.max(1, width / columns)
      const mid = height / 2

      for (let column = 0; column < columns; column += 1) {
        const amplitude = values[column] ?? 0
        // A floor keeps near-silence visible as a thin line rather than nothing.
        const barHeight = Math.max(1.5, amplitude * (height - 4))
        ctx.fillStyle = column <= playedTo ? PLAYED : BASE
        ctx.fillRect(column * barWidth, mid - barHeight / 2, Math.max(1, barWidth - 0.6), barHeight)
      }

      if (progress > 0 && progress < 1) {
        ctx.fillStyle = PLAYED
        ctx.fillRect(playedTo * barWidth, 0, 1, height)
      }
    },
    [width],
  )

  useEffect(() => {
    draw(peaks, playback.progress)
  }, [draw, peaks, playback.progress])

  const onSeek = useCallback(
    (event: React.MouseEvent<HTMLDivElement>) => {
      if (!peaks || disabled) return
      const bounds = event.currentTarget.getBoundingClientRect()
      // A hidden or not-yet-laid-out track has zero width; dividing by it would
      // poison the playhead offset with NaN for the rest of the clip's life.
      if (bounds.width <= 0) return
      const fraction = (event.clientX - bounds.left) / bounds.width
      void seek(fraction)
    },
    [peaks, disabled, seek],
  )

  const seconds = playback.duration || durationS || 0

  return (
    <div className="flex items-center gap-3">
      <button
        type="button"
        onClick={() => void toggle()}
        disabled={disabled || !url}
        aria-label={playback.playing ? 'Pause recording' : 'Play recording'}
        title={playback.playing ? 'Pause' : 'Play'}
        className="grid size-8 shrink-0 place-items-center rounded-full border border-ink-700 bg-ink-800 text-ink-200 transition hover:border-accent-500/50 hover:text-accent-400 disabled:cursor-not-allowed disabled:opacity-40"
      >
        {playback.playing ? (
          <svg viewBox="0 0 12 12" className="size-3" aria-hidden="true">
            <rect x="2" y="1.5" width="3" height="9" rx="1" fill="currentColor" />
            <rect x="7" y="1.5" width="3" height="9" rx="1" fill="currentColor" />
          </svg>
        ) : (
          <svg viewBox="0 0 12 12" className="size-3" aria-hidden="true">
            <path d="M3 1.5v9l7.5-4.5z" fill="currentColor" />
          </svg>
        )}
      </button>

      <div
        ref={wrapRef}
        role="slider"
        tabIndex={disabled || !peaks ? -1 : 0}
        aria-label="Seek within recording"
        aria-valuemin={0}
        aria-valuemax={Math.round(seconds * 100) / 100}
        aria-valuenow={Math.round(playback.position * 100) / 100}
        aria-valuetext={`${playback.position.toFixed(2)} of ${seconds.toFixed(2)} seconds`}
        onClick={onSeek}
        onKeyDown={(event) => {
          if (!peaks || disabled) return
          if (event.key === 'ArrowRight') void seek(playback.progress + 0.05)
          else if (event.key === 'ArrowLeft') void seek(playback.progress - 0.05)
          else return
          event.preventDefault()
        }}
        className={`min-w-0 flex-1 ${peaks && !disabled ? 'cursor-pointer' : ''}`}
      >
        <canvas ref={canvasRef} className="block h-14 w-full" />
      </div>

      <span className="w-14 shrink-0 text-right font-mono text-sm text-ink-400 tabular-nums">
        {formatDuration(playback.position || seconds)}
      </span>

      {playback.error ? (
        <span className="text-sm text-rose-300" role="alert">
          {playback.error}
        </span>
      ) : null}
    </div>
  )
}
