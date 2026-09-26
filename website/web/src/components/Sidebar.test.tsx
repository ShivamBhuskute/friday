import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { Sidebar } from './Sidebar'
import type { Health, IngestStats, Stats, SystemStatus } from '../lib/types'

const HEALTH: Health = {
  status: 'ok',
  ingest: 'listening',
  stt: 'ready',
  llm: 'ready',
  turns: 12,
  device_connected: true,
}

const STATS: Stats = { total: 12, done: 11, errors: 1, avg_transcript_ms: 102, avg_llm_ms: 1980 }

const INGEST: IngestStats = {
  listening: true,
  port: 9000,
  connections: 1,
  queue_depth: 0,
  bytes_received: 65_536,
  utterances_received: 3,
  last_audio_at: null,
  stt_device: 'cuda:0',
  stt_compute_type: 'int8_float16',
}

const SYSTEM: SystemStatus = {
  host: {
    load_1m: 1.75,
    mem_total_gb: 30.0,
    mem_used_gb: 9.5,
    disk_total_gb: 250,
    disk_free_gb: 125,
  },
  device: { free_heap: 120_000, rssi: -58 },
}

describe('Sidebar', () => {
  it('renders pipeline counters', () => {
    render(<Sidebar health={HEALTH} ingest={INGEST} stats={STATS} system={null} />)
    expect(screen.getByText('transitions').nextSibling).toHaveTextContent('11')
    expect(screen.getByText('failures').nextSibling).toHaveTextContent('1')
    expect(screen.getByText('stored').nextSibling).toHaveTextContent('12')
  })

  it('says the listener is down rather than hiding the problem', () => {
    render(
      <Sidebar health={HEALTH} ingest={{ ...INGEST, listening: false }} stats={STATS} system={null} />,
    )
    expect(screen.getByText('listener').nextSibling).toHaveTextContent('down')
  })

  it('formats byte counts and the STT compute type', () => {
    render(<Sidebar health={HEALTH} ingest={INGEST} stats={STATS} system={null} />)
    expect(screen.getByText('received').nextSibling).toHaveTextContent('64 KB')
    expect(screen.getByText('compute').nextSibling).toHaveTextContent('int8_float16')
  })

  it('shows memory and disk usage', () => {
    render(<Sidebar health={HEALTH} ingest={INGEST} stats={STATS} system={SYSTEM} />)
    expect(screen.getByText('memory').parentElement).toHaveTextContent('9.5 / 30.0 GB')
  })

  it('explains a missing device snapshot instead of rendering blanks', () => {
    render(
      <Sidebar
        health={HEALTH}
        ingest={INGEST}
        stats={STATS}
        system={{ host: {}, device: null, device_note: 'no sysmon snapshot pushed yet' }}
      />,
    )
    expect(screen.getByText(/no sysmon snapshot/)).toBeInTheDocument()
  })

  it('degrades to dashes when the server has not answered', () => {
    render(<Sidebar health={null} ingest={null} stats={null} system={null} />)
    expect(screen.getAllByText('--').length).toBeGreaterThan(3)
  })
})
