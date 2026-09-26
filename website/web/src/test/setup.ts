import '@testing-library/jest-dom/vitest'
import { afterEach, vi } from 'vitest'
import { cleanup } from '@testing-library/react'

// React 19 only treats `act()` as meaningful when the environment opts in.
;(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

/*
 * jsdom has no Web Audio, and the waveform decodes audio on mount. The stub
 * below is deliberately the *smallest* thing that satisfies the hook: a
 * mono-ish buffer of silence. Tests that care about playback stub the source
 * node themselves.
 */
class FakeAudioBuffer {
  readonly numberOfChannels = 1
  readonly length = 16_000
  readonly sampleRate = 16_000
  readonly duration = 1
  getChannelData(): Float32Array {
    return new Float32Array(this.length)
  }
}

class FakeAudioContext {
  state: AudioContextState = 'running'
  currentTime = 0
  destination = {}
  decodeAudioData(): Promise<AudioBuffer> {
    return Promise.resolve(new FakeAudioBuffer() as unknown as AudioBuffer)
  }
  createBufferSource(): AudioBufferSourceNode {
    return {
      buffer: null,
      connect: () => undefined,
      start: () => undefined,
      stop: () => undefined,
      disconnect: () => undefined,
      onended: null,
    } as unknown as AudioBufferSourceNode
  }
  resume(): Promise<void> {
    this.state = 'running'
    return Promise.resolve()
  }
  close(): Promise<void> {
    return Promise.resolve()
  }
}

Object.defineProperty(window, 'AudioContext', {
  value: FakeAudioContext,
  writable: true,
  configurable: true,
})

if (typeof globalThis.requestAnimationFrame !== 'function') {
  // jsdom omits rAF only in some environments; a timer is close enough for
  // asserting on rendered output.
  globalThis.requestAnimationFrame = ((cb: FrameRequestCallback) =>
    setTimeout(() => cb(0), 16)) as unknown as typeof requestAnimationFrame
  globalThis.cancelAnimationFrame = ((handle: number) =>
    clearTimeout(handle as unknown as ReturnType<typeof setTimeout>)) as typeof cancelAnimationFrame
}
