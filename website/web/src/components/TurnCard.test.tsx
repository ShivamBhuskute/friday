import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { TurnCard } from './TurnCard'
import { makeToolCall, makeTurn } from '../test/factories'

describe('TurnCard', () => {
  it('shows the instruction and the answer together', () => {
    render(<TurnCard turn={makeTurn()} onDelete={vi.fn()} />)
    expect(screen.getByText(/weather in Pune/)).toBeInTheDocument()
    expect(screen.getByTestId('turn-answer')).toHaveTextContent('It is 28 degrees Celsius in Pune.')
  })

  it('labels the pipeline stage on in-flight turns', () => {
    render(<TurnCard turn={makeTurn({ state: 'transcribing' })} onDelete={vi.fn()} />)
    const pill = screen.getByTestId('status-pill')
    expect(pill).toHaveAttribute('data-state', 'transcribing')
    expect(pill).toHaveTextContent('transcribing')
  })

  it('says it is listening when there is no transcript yet', () => {
    render(
      <TurnCard
        turn={makeTurn({ state: 'uploaded', transcript: null, answer: null, confidence: null })}
        onDelete={vi.fn()}
      />,
    )
    expect(screen.getByText(/listening/)).toBeInTheDocument()
  })

  it('shows the failure reason rather than a silent empty card', () => {
    render(
      <TurnCard
        turn={makeTurn({ state: 'error', answer: null, error: 'transcription crashed' })}
        onDelete={vi.fn()}
      />,
    )
    expect(screen.getByTestId('turn-error')).toHaveTextContent('transcription crashed')
    expect(screen.queryByTestId('turn-answer')).not.toBeInTheDocument()
  })

  it('surfaces the recognition confidence', () => {
    render(<TurnCard turn={makeTurn({ confidence: 0.5 })} onDelete={vi.fn()} />)
    expect(screen.getByText('50%')).toBeInTheDocument()
  })

  it('omits confidence when the model produced none', () => {
    render(<TurnCard turn={makeTurn({ confidence: null })} onDelete={vi.fn()} />)
    expect(screen.queryByTitle('Speech recognition confidence')).not.toBeInTheDocument()
  })

  it('reports per-stage latency and the clip format', () => {
    render(
      <TurnCard
        turn={makeTurn({ duration_s: 2.75, sample_rate: 16_000, channels: 1 })}
        onDelete={vi.fn()}
      />,
    )
    expect(screen.getByTitle('Time spent transcribing')).toHaveTextContent('104ms')
    expect(screen.getByTitle(/including tool calls/)).toHaveTextContent('2.21s')
    expect(screen.getByText('audio 2.75s')).toBeInTheDocument()
    expect(screen.getByText('16kHz mono')).toBeInTheDocument()
  })

  it('leaves the timings blank when a stage never ran', () => {
    render(<TurnCard turn={makeTurn({ transcript_ms: null, llm_ms: null })} onDelete={vi.fn()} />)
    expect(screen.getByTitle('Time spent transcribing')).toHaveTextContent('--')
  })

  it('hides the tool trace when the model called nothing', () => {
    render(<TurnCard turn={makeTurn()} onDelete={vi.fn()} />)
    expect(screen.queryByText(/tools?$/)).not.toBeInTheDocument()
  })

  it('collapses the tool trace until asked', async () => {
    const user = userEvent.setup()
    render(<TurnCard turn={makeTurn({ tool_calls: [makeToolCall()] })} onDelete={vi.fn()} />)

    const toggle = screen.getByRole('button', { expanded: false })
    expect(toggle).toHaveTextContent('1 tool')
    // Arguments and results are the diagnosis, so they stay hidden by default.
    expect(screen.queryByText(/city=/)).not.toBeInTheDocument()

    await user.click(toggle)
    expect(screen.getByText(/city="Pune"/)).toBeInTheDocument()
    expect(screen.getByText('28 C, clear')).toBeInTheDocument()
  })

  it('chains multiple tool calls in order', () => {
    render(
      <TurnCard
        turn={makeTurn({
          tool_calls: [makeToolCall(), makeToolCall({ name: 'calculate', duration_ms: 12 })],
        })}
        onDelete={vi.fn()}
      />,
    )
    expect(screen.getByText(/get_weather -> calculate/)).toBeInTheDocument()
  })

  it('marks a trace that failed', () => {
    render(
      <TurnCard
        turn={makeTurn({
          tool_calls: [makeToolCall({ error: 'upstream timeout', result: null })],
        })}
        onDelete={vi.fn()}
      />,
    )
    expect(screen.getByText(/failed/)).toBeInTheDocument()
  })

  it('offers playback only when there is audio', () => {
    // The waveform fetches on mount; hold the response open so the assertion is
    // about the controls, not about a decode landing mid-test.
    vi.stubGlobal('fetch', () => new Promise(() => {}))
    const { rerender } = render(<TurnCard turn={makeTurn()} onDelete={vi.fn()} />)
    expect(screen.queryByRole('button', { name: /recording/ })).not.toBeInTheDocument()

    rerender(
      <TurnCard
        turn={makeTurn({ audio_url: '/api/audio/turn-0001', duration_s: 1.4 })}
        onDelete={vi.fn()}
      />,
    )
    expect(screen.getByRole('button', { name: 'Play recording' })).toBeInTheDocument()
  })

  it('deletes the turn when asked', async () => {
    const user = userEvent.setup()
    const onDelete = vi.fn()
    render(<TurnCard turn={makeTurn()} onDelete={onDelete} />)
    await user.click(screen.getByRole('button', { name: 'Delete this turn' }))
    expect(onDelete).toHaveBeenCalledWith('turn-0001')
  })
})

describe('TurnCard: the newest turn is the highlight', () => {
  it('marks itself as latest when told to', () => {
    render(<TurnCard turn={makeTurn()} onDelete={vi.fn()} latest />)
    expect(screen.getByTestId('turn-card')).toHaveAttribute('data-latest', 'true')
  })

  it('does not claim to be latest by default', () => {
    render(<TurnCard turn={makeTurn()} onDelete={vi.fn()} />)
    expect(screen.getByTestId('turn-card')).toHaveAttribute('data-latest', 'false')
  })

  it('sets the instruction larger on the latest card than on an older one', () => {
    // The whole point of the prop: at a glance the room can tell which
    // exchange just happened, in a column of history.
    const { unmount } = render(<TurnCard turn={makeTurn()} onDelete={vi.fn()} latest />)
    const big = screen.getByTestId('turn-instruction').className
    unmount()

    render(<TurnCard turn={makeTurn()} onDelete={vi.fn()} />)
    const small = screen.getByTestId('turn-instruction').className

    expect(big).toContain('text-2xl')
    expect(small).toContain('text-base')
    expect(small).not.toContain('text-2xl')
  })

  it('enlarges the answer too, so the reply is readable from the back of a room', () => {
    const { unmount } = render(<TurnCard turn={makeTurn()} onDelete={vi.fn()} latest />)
    const big = screen.getByTestId('turn-answer').className
    unmount()

    render(<TurnCard turn={makeTurn()} onDelete={vi.fn()} />)
    expect(big).toContain('text-xl')
    expect(screen.getByTestId('turn-answer').className).toContain('text-base')
  })

  it('says "latest" so the highlight is not only visual', () => {
    render(<TurnCard turn={makeTurn()} onDelete={vi.fn()} latest />)
    expect(screen.getByText(/latest/i)).toBeInTheDocument()
  })

  it('does not shout "latest" on an older card', () => {
    render(<TurnCard turn={makeTurn()} onDelete={vi.fn()} />)
    expect(screen.queryByText(/latest/i)).not.toBeInTheDocument()
  })
})
