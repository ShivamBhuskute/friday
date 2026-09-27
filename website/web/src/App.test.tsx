import { act, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import App from './App'
import { makeTurn } from './test/factories'

const META = {
  health: {
    status: 'ok',
    ingest: 'listening',
    stt: 'ready',
    llm: 'ready',
    turns: 1,
    device_connected: true,
  },
  ingest: {
    listening: true,
    port: 9000,
    connections: 0,
    queue_depth: 0,
    bytes_received: 0,
    utterances_received: 0,
    last_audio_at: null,
  },
  stats: { total: 1, done: 1, errors: 0, avg_transcript_ms: 100, avg_llm_ms: 900 },
  system: { host: {}, device: null },
}

interface Routes {
  turns?: unknown[]
  ask?: (text: string) => unknown
  /** HTTP status the text-turn endpoint should return. */
  askStatus?: number
}

/** Wire a fake REST surface plus a WebSocket the test can push frames on. */
function stub(routes: Routes = {}) {
  const turns = routes.turns ?? [makeTurn()]
  const handlers: Record<string, (init?: RequestInit) => unknown> = {
    '/api/turns': (init) => {
      if (init?.method === 'POST') {
        const body = JSON.parse(String(init.body)) as { text: string }
        return routes.ask?.(body.text) ?? makeTurn({ id: 'turn-new', transcript: body.text })
      }
      return turns
    },
    '/api/health': () => META.health,
    '/api/ingest/stats': () => META.ingest,
    '/api/stats': () => META.stats,
    '/api/system': () => META.system,
  }

  const fetchSpy = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input)
    const key = Object.keys(handlers).find((k) => url === k || url.startsWith(`${k}?`))
    if (key === undefined) throw new TypeError(`unexpected fetch: ${url}`)
    const status = init?.method === 'POST' ? (routes.askStatus ?? 201) : 200
    return {
      ok: status < 400,
      status,
      statusText: status < 400 ? 'OK' : 'Internal Server Error',
      json: async () => handlers[key]!(init),
      arrayBuffer: async () => new ArrayBuffer(8),
    }
  })
  vi.stubGlobal('fetch', fetchSpy)

  class FakeSocket {
    static instances: FakeSocket[] = []
    readyState = 0
    onopen: (() => void) | null = null
    onclose: (() => void) | null = null
    onerror: (() => void) | null = null
    onmessage: ((event: { data: string }) => void) | null = null
    constructor(public url: string) {
      FakeSocket.instances.push(this)
      setTimeout(() => this.onopen?.(), 0)
    }
    send(): void {}
    close(): void {
      this.readyState = 3
      this.onclose?.()
    }
    /** Push a server frame to the app. */
    emit(payload: unknown): void {
      this.onmessage?.({ data: JSON.stringify(payload) })
    }
  }
  vi.stubGlobal('WebSocket', FakeSocket)
  return { fetchSpy, FakeSocket, turns }
}

describe('App', () => {
  it('renders the turns the server already had', async () => {
    stub()
    render(<App />)
    expect(await screen.findByText(/weather in Pune/)).toBeInTheDocument()
  })

  it('shows an empty state with the replay hint when there is nothing yet', async () => {
    stub({ turns: [] })
    render(<App />)
    expect(await screen.findByText('No turns yet.')).toBeInTheDocument()
    expect(screen.getByText(/replay_device\.py/)).toBeInTheDocument()
  })

  it('reports the feed as live once the socket opens', async () => {
    stub()
    render(<App />)
    await waitFor(() => {
      expect(screen.getByTitle('WebSocket connection to the turn feed')).toHaveTextContent(
        'feed live',
      )
    })
  })

  it('updates a card in place when the feed pushes a state change', async () => {
    const { FakeSocket } = stub({ turns: [makeTurn({ state: 'thinking', answer: null })] })
    render(<App />)

    const card = await screen.findByTestId('turn-card')
    expect(card).toHaveAttribute('data-state', 'thinking')

    const socket = await waitFor(() => {
      const found = FakeSocket.instances[0]
      if (!found) throw new Error('no socket yet')
      return found
    })
    act(() => {
      socket.emit({ event: 'turn.updated', turn: makeTurn({ answer: '28 degrees Celsius.' }) })
    })

    await waitFor(() => {
      expect(screen.getByTestId('turn-card')).toHaveAttribute('data-state', 'done')
    })
    expect(screen.getByTestId('turn-answer')).toHaveTextContent('28 degrees Celsius.')
  })

  it('prepends a brand new turn pushed by the feed', async () => {
    const { FakeSocket } = stub({ turns: [makeTurn({ seq: 5, id: 'old' })] })
    render(<App />)
    const list = await screen.findByTestId('turn-list')

    const socket = await waitFor(() => {
      const found = FakeSocket.instances[0]
      if (!found) throw new Error('no socket yet')
      return found
    })
    act(() => {
      socket.emit({
        event: 'turn.created',
        turn: makeTurn({ seq: 6, id: 'new', transcript: 'what is 7 times 23' }),
      })
    })

    await waitFor(() => {
      expect(screen.getByText('what is 7 times 23')).toBeInTheDocument()
    })
    expect(list.children).toHaveLength(2)
    // Newest first.
    expect(list.children[0]).toHaveTextContent('what is 7 times 23')
  })

  it('ignores a malformed frame instead of crashing the console', async () => {
    const { FakeSocket } = stub()
    render(<App />)
    const socket = await waitFor(() => {
      const found = FakeSocket.instances[0]
      if (!found) throw new Error('no socket yet')
      return found
    })
    act(() => {
      socket.onmessage?.({ data: 'not json' })
      socket.emit({ event: 'something-else' })
    })
    expect(screen.getByTestId('turn-card')).toBeInTheDocument()
  })

  it('sends a typed question through the same pipeline', async () => {
    const user = userEvent.setup()
    const asked: string[] = []
    stub({ ask: (text) => { asked.push(text); return makeTurn({ id: 'typed', transcript: text }) } })
    render(<App />)

    const input = await screen.findByLabelText('Ask FRIDAY a question')
    await user.type(input, 'what is 7 times 23')
    await user.click(screen.getByRole('button', { name: 'send' }))

    await waitFor(() => expect(asked).toEqual(['what is 7 times 23']))
    expect(await screen.findByText('what is 7 times 23')).toBeInTheDocument()
    // The field is cleared so the next question can be typed straight away.
    expect(input).toHaveValue('')
  })

  it('will not submit an empty question', async () => {
    const user = userEvent.setup()
    const { fetchSpy } = stub()
    render(<App />)
    await screen.findByLabelText('Ask FRIDAY a question')
    expect(screen.getByRole('button', { name: 'send' })).toBeDisabled()
    await user.type(screen.getByLabelText('Ask FRIDAY a question'), '   ')
    expect(screen.getByRole('button', { name: 'send' })).toBeDisabled()
    expect(fetchSpy).not.toHaveBeenCalledWith('/api/turns', expect.objectContaining({ method: 'POST' }))
  })

  it('keeps the typed text and explains the failure when the server rejects it', async () => {
    const user = userEvent.setup()
    stub({ askStatus: 500 })
    render(<App />)

    const input = await screen.findByLabelText('Ask FRIDAY a question')
    await user.type(input, 'hello friday')
    await user.click(screen.getByRole('button', { name: 'send' }))

    expect(await screen.findByRole('alert')).toHaveTextContent(/500/)
    expect(input).toHaveValue('hello friday')
  })

  it('leaves the rest of the console usable after a failed send', async () => {
    const user = userEvent.setup()
    stub({ askStatus: 500 })
    render(<App />)
    await user.type(await screen.findByLabelText('Ask FRIDAY a question'), 'hello friday')
    await user.click(screen.getByRole('button', { name: 'send' }))
    await screen.findByRole('alert')
    // No offline banner: a rejected turn is not a dead server.
    expect(screen.queryByText(/cannot reach the FRIDAY server/)).not.toBeInTheDocument()
  })

  it('surfaces an offline server with a retry affordance', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => {
        throw new TypeError('Failed to fetch')
      }),
    )
    class DeadSocket {
      constructor() {
        setTimeout(() => this.onclose?.(), 0)
      }
      onclose: (() => void) | null = null
      close(): void {}
    }
    vi.stubGlobal('WebSocket', DeadSocket)

    render(<App />)
    const banner = await screen.findByRole('alert')
    expect(banner).toHaveTextContent(/cannot reach the FRIDAY server/)
    expect(screen.getByRole('button', { name: 'retry' })).toBeInTheDocument()
  })
})

describe('App: highlighting the newest turn', () => {
  it('highlights only the most recent turn in a list of several', async () => {
    const { FakeSocket } = stub({
      turns: [
        makeTurn({ id: 'newest', seq: 9, transcript: 'the newest instruction' }),
        makeTurn({ id: 'middle', seq: 8, transcript: 'an older instruction' }),
        makeTurn({ id: 'oldest', seq: 7, transcript: 'the oldest instruction' }),
      ],
    })

    render(<App />)
    await screen.findByText('the newest instruction')

    const cards = screen.getAllByTestId('turn-card')
    expect(cards).toHaveLength(3)

    // `turns` is newest-first, so the first card is the one to emphasise.
    expect(cards[0]).toHaveAttribute('data-latest', 'true')
    expect(cards[1]).toHaveAttribute('data-latest', 'false')
    expect(cards[2]).toHaveAttribute('data-latest', 'false')
    expect(FakeSocket).toBeDefined()
  })

  it('moves the highlight when a newer turn arrives over the socket', async () => {
    const { FakeSocket } = stub({ turns: [makeTurn({ id: 'first', seq: 1 })] })

    render(<App />)
    await screen.findByText(/weather in Pune/)
    expect(screen.getAllByTestId('turn-card')[0]).toHaveAttribute('data-latest', 'true')

    const socket = await waitFor(() => {
      const found = FakeSocket.instances[0]
      if (!found) throw new Error('no socket yet')
      return found
    })
    act(() => {
      socket.emit({
        event: 'turn.created',
        turn: makeTurn({ id: 'second', seq: 2, transcript: 'a brand new question' }),
      })
    })

    await screen.findByText('a brand new question')
    const cards = screen.getAllByTestId('turn-card')
    expect(cards[0]).toHaveAttribute('data-latest', 'true')
    expect(cards[0]).toHaveTextContent('a brand new question')
    expect(cards[1]).toHaveAttribute('data-latest', 'false')
  })
})
