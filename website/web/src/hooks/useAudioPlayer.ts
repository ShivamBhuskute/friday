import { useCallback, useEffect, useRef, useState } from 'react'
import { computePeaks, toMono, type Peaks } from '../lib/waveform'

/**
 * Decode a WAV once, cache it, and drive playback from a single AudioContext.
 *
 * Two things make this awkward in the raw DOM. `decodeAudioData` is async and
 * re-decoding on every play stutters, so the buffer is memoised. And
 * `AudioBufferSourceNode` cannot be paused or seeked, so playback position is
 * tracked from `AudioContext.currentTime` minus the offset the source started
 * at, with `stop()` + a fresh start for pause and seek.
 */

export type AudioStatus = 'idle' | 'loading' | 'ready' | 'error'

/** Enough resolution for a 300px-wide element at 1x, and cheap to draw. */
const PEAK_COLUMNS = 240

interface Playback {
  /** 0..1 through the clip. */
  progress: number
  /** Position in seconds. */
  position: number
  /** Clip length in seconds, 0 until the buffer is decoded. */
  duration: number
  status: AudioStatus
  playing: boolean
  error: string | null
}

const IDLE: Playback = {
  progress: 0,
  position: 0,
  duration: 0,
  status: 'idle',
  playing: false,
  error: null,
}

/** One AudioContext per document; browsers cap how many may exist. */
let sharedContext: AudioContext | null = null

function audioContext(): AudioContext {
  if (sharedContext === null) {
    const Ctor =
      window.AudioContext ??
      (window as unknown as { webkitAudioContext?: typeof AudioContext })
        .webkitAudioContext
    if (!Ctor) throw new Error('this browser has no Web Audio support')
    sharedContext = new Ctor()
  }
  return sharedContext
}

/**
 * Decodes run one at a time.
 *
 * Fifty cards mounting at once would otherwise issue fifty parallel
 * fetch+decode pairs, and the main thread stalls long enough to be visible as a
 * hitch when the page first paints. The clips are small, so a serial queue
 * fills in the waveforms within a frame or two either way.
 */
let decodeChain: Promise<unknown> = Promise.resolve()

function enqueueDecode<T>(task: () => Promise<T>): Promise<T> {
  const next = decodeChain.then(task, task)
  // Keep the chain alive after a failure; one bad clip must not wedge the rest.
  decodeChain = next.then(
    () => undefined,
    () => undefined,
  )
  return next
}

export function useAudioPlayer(url: string | null) {
  const [playback, setPlayback] = useState<Playback>(IDLE)
  const [peaks, setPeaks] = useState<Peaks | null>(null)

  const contextRef = useRef<AudioContext | null>(null)
  const bufferRef = useRef<AudioBuffer | null>(null)
  const sourceRef = useRef<AudioBufferSourceNode | null>(null)
  const startedAtRef = useRef(0)
  const offsetRef = useRef(0)
  const rafRef = useRef(0)

  const stopSource = useCallback(() => {
    const source = sourceRef.current
    if (source) {
      source.onended = null
      try {
        source.stop()
      } catch {
        // Already stopped; nothing to do.
      }
      source.disconnect()
      sourceRef.current = null
    }
    cancelAnimationFrame(rafRef.current)
    rafRef.current = 0
  }, [])

  /** Release the context, the buffer and the object URL on unmount. */
  useEffect(
    () => () => {
      stopSource()
      bufferRef.current = null
      void contextRef.current?.close().catch(() => undefined)
      contextRef.current = null
    },
    [stopSource],
  )

  // A new URL means a new recording: throw away everything cached for the old one.
  useEffect(() => {
    stopSource()
    bufferRef.current = null
    setPeaks(null)
    setPlayback(IDLE)
  }, [url, stopSource])

  const load = useCallback(async (): Promise<AudioBuffer> => {
    if (bufferRef.current) return bufferRef.current
    if (!url) throw new Error('no audio for this turn')

    setPlayback((prev) => ({ ...prev, status: 'loading', error: null }))
    try {
      return await enqueueDecode(async () => {
        // A second caller may have finished the load while we queued.
        if (bufferRef.current) return bufferRef.current
        const context = contextRef.current ?? audioContext()
        contextRef.current = context

        const response = await fetch(url)
        if (!response.ok) {
          throw new Error(`audio unavailable (${response.status})`)
        }
        const bytes = await response.arrayBuffer()
        const buffer = await context.decodeAudioData(bytes)
        bufferRef.current = buffer
        setPeaks(computePeaks(toMono(buffer), buffer.sampleRate, PEAK_COLUMNS))
        setPlayback({
          progress: 0,
          position: 0,
          duration: buffer.duration,
          status: 'ready',
          playing: false,
          error: null,
        })
        return buffer
      })
    } catch (cause) {
      const message =
        cause instanceof Error ? cause.message : 'could not decode this recording'
      setPlayback((prev) => ({ ...prev, status: 'error', error: message }))
      throw cause
    }
  }, [url])

  // Draw the waveform as soon as the card appears: playback is the point of the
  // row, and waiting for a click to reveal the shape defeats the purpose.
  useEffect(() => {
    if (!url) return
    void load().catch(() => undefined)
  }, [load, url])

  /** Advance the playhead. Driven by rAF, not by the source's `onended`. */
  const tick = useCallback(() => {
    const buffer = bufferRef.current
    if (!buffer || !sourceRef.current) return
    const position = offsetRef.current + (contextRef.current?.currentTime ?? 0) - startedAtRef.current
    const clamped = Math.max(0, Math.min(buffer.duration, position))
    setPlayback((prev) => ({
      ...prev,
      position: clamped,
      progress: buffer.duration > 0 ? clamped / buffer.duration : 0,
    }))
    rafRef.current = requestAnimationFrame(tick)
  }, [])

  const play = useCallback(async () => {
    const buffer = await load()
    const context = contextRef.current
    if (!context) return
    // Autoplay policy: a context created outside a gesture starts suspended.
    if (context.state === 'suspended') await context.resume()

    stopSource()
    const source = context.createBufferSource()
    source.buffer = buffer
    source.connect(context.destination)
    source.onended = () => {
      // Fires on natural end and on our own stop(); only reset on the former.
      if (sourceRef.current !== source) return
      stopSource()
      offsetRef.current = 0
      setPlayback((prev) => ({ ...prev, playing: false, position: 0, progress: 0 }))
    }
    const from = offsetRef.current >= buffer.duration ? 0 : offsetRef.current
    source.start(0, from)
    sourceRef.current = source
    startedAtRef.current = context.currentTime
    offsetRef.current = from
    setPlayback((prev) => ({ ...prev, playing: true, error: null }))
    rafRef.current = requestAnimationFrame(tick)
  }, [load, stopSource, tick])

  const pause = useCallback(() => {
    const buffer = bufferRef.current
    const context = contextRef.current
    if (!buffer || !context) return
    const position = offsetRef.current + context.currentTime - startedAtRef.current
    offsetRef.current = Math.max(0, Math.min(buffer.duration, position))
    stopSource()
    setPlayback((prev) => ({ ...prev, playing: false }))
  }, [stopSource])

  const toggle = useCallback(() => {
    if (playback.playing) pause()
    else void play()
  }, [pause, play, playback.playing])

  const seek = useCallback(
    async (fraction: number) => {
      const buffer = await load()
      const clamped = Math.max(0, Math.min(1, fraction))
      offsetRef.current = clamped * buffer.duration
      setPlayback((prev) => ({ ...prev, position: offsetRef.current, progress: clamped }))
      if (playback.playing) {
        // Restart at the new offset: a BufferSource cannot be moved.
        stopSource()
        offsetRef.current = clamped * buffer.duration
        const context = contextRef.current
        const source = context?.createBufferSource()
        if (!context || !source) return
        source.buffer = buffer
        source.connect(context.destination)
        source.onended = () => {
          if (sourceRef.current !== source) return
          stopSource()
          offsetRef.current = 0
          setPlayback((prev) => ({ ...prev, playing: false, position: 0, progress: 0 }))
        }
        source.start(0, offsetRef.current)
        sourceRef.current = source
        startedAtRef.current = context.currentTime
        rafRef.current = requestAnimationFrame(tick)
      }
    },
    [load, playback.playing, stopSource, tick],
  )

  return { playback, peaks, play, pause, toggle, seek, load }
}
