import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { Waveform } from './Waveform'

function mockFetchOnce(bytes: ArrayBuffer, ok = true) {
  const spy = vi.fn(async () => ({
    ok,
    status: ok ? 200 : 404,
    arrayBuffer: async () => bytes,
  }))
  vi.stubGlobal('fetch', spy)
  return spy
}

describe('Waveform', () => {
  it('disables the transport when there is no audio to play', () => {
    render(<Waveform url={null} />)
    expect(screen.getByRole('button', { name: 'Play recording' })).toBeDisabled()
  })

  it('fetches the clip on mount, without waiting for a click', async () => {
    const fetchSpy = mockFetchOnce(new ArrayBuffer(128))
    render(<Waveform url="/api/audio/abc" durationS={1.4} />)
    await waitFor(() => expect(fetchSpy).toHaveBeenCalledWith('/api/audio/abc'))
    await waitFor(() => {
      expect(screen.getByRole('slider', { name: /seek/i })).toHaveAttribute('tabindex', '0')
    })
  })

  it('loads the clip and becomes seekable', async () => {
    mockFetchOnce(new ArrayBuffer(128))
    render(<Waveform url="/api/audio/abc" durationS={1.4} />)

    await waitFor(() => {
      expect(screen.getByRole('slider', { name: /seek/i })).toHaveAttribute('tabindex', '0')
    })
  })

  it('toggles the transport label between play and pause', async () => {
    const user = userEvent.setup()
    mockFetchOnce(new ArrayBuffer(128))
    render(<Waveform url="/api/audio/abc" />)

    const play = await screen.findByRole('button', { name: 'Play recording' })
    await user.click(play)
    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Pause recording' })).toBeInTheDocument()
    })
  })

  it('reports a failed fetch instead of silently doing nothing', async () => {
    mockFetchOnce(new ArrayBuffer(0), false)
    render(<Waveform url="/api/audio/abc" />)
    expect(await screen.findByRole('alert')).toHaveTextContent(/audio unavailable/i)
  })

  it('refetches nothing on a second play: the buffer is memoised', async () => {
    const user = userEvent.setup()
    const fetchSpy = mockFetchOnce(new ArrayBuffer(128))
    render(<Waveform url="/api/audio/abc" />)

    const play = await screen.findByRole('button', { name: 'Play recording' })
    await user.click(play)
    await user.click(screen.getByRole('button', { name: 'Pause recording' }))
    await user.click(screen.getByRole('button', { name: 'Play recording' }))

    expect(fetchSpy).toHaveBeenCalledTimes(1)
  })

  it('surfaces a decoder failure as an error, not a blank box', async () => {
    // Break decoding only: the transport still resolves.
    const Ctor = window.AudioContext as unknown as {
      prototype: { decodeAudioData: (bytes: ArrayBuffer) => Promise<AudioBuffer> }
    }
    const spy = vi
      .spyOn(Ctor.prototype, 'decodeAudioData')
      .mockRejectedValue(new Error('not a wav'))
    mockFetchOnce(new ArrayBuffer(128))

    render(<Waveform url="/api/audio/abc" />)
    expect(await screen.findByRole('alert')).toHaveTextContent(/not a wav/i)
    // The row must not offer seeking on a clip that never decoded.
    expect(screen.getByRole('slider', { name: /seek/i })).toHaveAttribute('tabindex', '-1')
    spy.mockRestore()
  })

  it('ignores a click on a zero-width track instead of seeking to NaN', async () => {
    const user = userEvent.setup()
    mockFetchOnce(new ArrayBuffer(128))
    render(<Waveform url="/api/audio/abc" />)

    const slider = await screen.findByRole('slider', { name: /seek/i })
    // jsdom reports a 0x0 rect, which is exactly the layout case the guard is for.
    await user.click(slider)

    expect(slider.getAttribute('aria-valuetext')).toBe('0.00 of 1.00 seconds')
    expect(slider.getAttribute('aria-valuenow')).not.toBe('NaN')
  })

  it('seeks to a fraction of the clip', async () => {
    mockFetchOnce(new ArrayBuffer(128))
    render(<Waveform url="/api/audio/abc" />)

    const slider = await screen.findByRole('slider', { name: /seek/i })
    // Give the track a real width so the click maps to a fraction.
    vi.spyOn(slider, 'getBoundingClientRect').mockReturnValue({
      left: 0,
      top: 0,
      width: 200,
      height: 56,
      right: 200,
      bottom: 56,
      x: 0,
      y: 0,
      toJSON: () => ({}),
    } as DOMRect)
    fireEvent.click(slider, { clientX: 100 })

    await waitFor(() => {
      expect(slider.getAttribute('aria-valuetext')).toBe('0.50 of 1.00 seconds')
    })
  })

  it('seeks with the arrow keys', async () => {
    const user = userEvent.setup()
    mockFetchOnce(new ArrayBuffer(128))
    render(<Waveform url="/api/audio/abc" />)

    const slider = await screen.findByRole('slider', { name: /seek/i })
    slider.focus()
    await user.keyboard('{ArrowRight}{ArrowRight}')
    await waitFor(() => {
      expect(Number(slider.getAttribute('aria-valuenow'))).toBeGreaterThan(0)
    })
  })

  it('seeks backwards with the left arrow', async () => {
    const user = userEvent.setup()
    mockFetchOnce(new ArrayBuffer(128))
    render(<Waveform url="/api/audio/abc" />)

    const slider = await screen.findByRole('slider', { name: /seek/i })
    slider.focus()
    await user.keyboard('{ArrowRight}{ArrowRight}{ArrowLeft}')
    const now = Number(slider.getAttribute('aria-valuenow'))
    expect(now).toBeGreaterThanOrEqual(0)
    expect(now).toBeLessThan(1)
  })
})
