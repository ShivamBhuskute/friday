/**
 * Wire types. These mirror `server/schemas.py` exactly; if you change one,
 * change the other, and `npm run typecheck` will only catch the frontend half.
 */

export type TurnState =
  | 'uploaded'
  | 'transcribing'
  | 'thinking'
  | 'calling'
  | 'done'
  | 'no_speech'
  | 'unclear'
  | 'error'

export const TERMINAL_STATES: readonly TurnState[] = [
  'done',
  'no_speech',
  'unclear',
  'error',
]

export function isTerminal(state: TurnState): boolean {
  return TERMINAL_STATES.includes(state)
}

export interface ToolCall {
  name: string
  arguments: Record<string, unknown>
  result: unknown
  error: string | null
  duration_ms: number | null
}

export interface Turn {
  id: string
  seq: number
  created_at: string
  updated_at: string
  state: TurnState
  audio_url: string | null
  duration_s: number | null
  sample_rate: number | null
  channels: number | null
  source: string | null
  transcript: string | null
  confidence: number | null
  answer: string | null
  error: string | null
  tool_calls: ToolCall[]
  transcript_ms: number | null
  llm_ms: number | null
}

export interface Health {
  status: 'ok' | 'degraded'
  ingest: 'listening' | 'down'
  /** 'ready' once the model is resident, 'lazy' until the first clip. */
  stt: 'ready' | 'lazy'
  llm: 'ready' | 'lazy'
  turns: number
  device_connected: boolean
  version?: string
}

export interface IngestStats {
  listening: boolean
  port: number
  connections: number
  queue_depth: number
  bytes_received: number
  utterances_received: number
  /** Unix seconds, or null before the first utterance. */
  last_audio_at: number | null
  stt_device?: string
  stt_compute_type?: string
  llm_loaded?: boolean
}

export interface HostStatus {
  cpu_percent?: number | null
  load_1m?: number | null
  mem_total_gb?: number | null
  mem_used_gb?: number | null
  disk_total_gb?: number | null
  disk_free_gb?: number | null
}

export interface SystemStatus {
  host: HostStatus
  device: Record<string, unknown> | null
  device_note?: string
}

export interface Stats {
  total: number
  done: number
  errors: number
  avg_transcript_ms: number | null
  avg_llm_ms: number | null
}

export interface WsEvent {
  event: 'turn.created' | 'turn.updated'
  turn: Turn
}

/** A turn plus client-only bookkeeping that the server does not send. */
export interface TurnView extends Turn {
  /** Set when a fetch for this turn's audio failed, so the UI can say why. */
  audioError?: string
}
