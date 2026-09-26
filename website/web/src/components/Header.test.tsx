import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { Header } from './Header'
import type { Health, IngestStats } from '../lib/types'

const HEALTHY: Health = {
  status: 'ok',
  ingest: 'listening',
  stt: 'ready',
  llm: 'ready',
  turns: 4,
  device_connected: true,
}

const STATS: IngestStats = {
  listening: true,
  port: 9000,
  connections: 1,
  queue_depth: 0,
  bytes_received: 32_000,
  utterances_received: 2,
  last_audio_at: null,
  stt_device: 'cuda',
  stt_compute_type: 'int8_float16',
}

describe('Header', () => {
  it('reports a fully healthy stack', () => {
    render(<Header socket="open" health={HEALTHY} ingest={STATS} />)
    expect(screen.getByTitle('WebSocket connection to the turn feed')).toHaveTextContent('feed live')
    expect(screen.getByTitle('Speech recognition model')).toHaveTextContent('int8_float16')
    expect(screen.getByTitle('Local language model')).toHaveTextContent('llm ready')
  })

  it('distinguishes "no device yet" from "the feed is broken"', () => {
    render(
      <Header
        socket="open"
        health={{ ...HEALTHY, device_connected: false }}
        ingest={STATS}
      />,
    )
    expect(screen.getByText(/device idle/)).toBeInTheDocument()
    expect(screen.getByText(/feed live/)).toBeInTheDocument()
  })

  it('flags a closed socket', () => {
    render(<Header socket="closed" health={null} ingest={null} />)
    expect(screen.getByTitle('WebSocket connection to the turn feed')).toHaveTextContent(
      'feed offline',
    )
  })

  it('shows the lazy model state before anything has been transcribed', () => {
    render(
      <Header socket="open" health={{ ...HEALTHY, stt: 'lazy', llm: 'lazy' }} ingest={null} />,
    )
    expect(screen.getByTitle('Speech recognition model')).toHaveTextContent('stt lazy')
    expect(screen.getByTitle('Local language model')).toHaveTextContent('llm lazy')
  })

  it('surfaces queue depth only when work is waiting', () => {
    const { rerender } = render(<Header socket="open" health={HEALTHY} ingest={STATS} />)
    expect(screen.queryByText(/queued/)).not.toBeInTheDocument()

    rerender(<Header socket="open" health={HEALTHY} ingest={{ ...STATS, queue_depth: 2 }} />)
    expect(screen.getByText('2 queued')).toBeInTheDocument()
  })
})
