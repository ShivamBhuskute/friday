import type { Health, IngestStats, Stats, SystemStatus, Turn, WsEvent } from './types'

/** Error carrying the HTTP status, so callers can distinguish 404 from 500. */
export class ApiError extends Error {
  readonly status: number
  constructor(status: number, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(path, {
      ...init,
      headers: { 'Content-Type': 'application/json', ...init?.headers },
    })
  } catch (cause) {
    // A refused connection means the server is down; say so plainly rather than
    // surfacing "fetch failed", which tells the user nothing.
    throw new ApiError(0, 'cannot reach the FRIDAY server')
  }
  if (!response.ok) {
    throw new ApiError(response.status, `${response.status} ${response.statusText}`)
  }
  if (response.status === 204) return undefined as T
  return (await response.json()) as T
}

export const api = {
  health: () => request<Health>('/api/health'),
  ingestStats: () => request<IngestStats>('/api/ingest/stats'),
  stats: () => request<Stats>('/api/stats'),
  system: () => request<SystemStatus>('/api/system'),
  turns: (limit = 50, offset = 0) =>
    request<Turn[]>(`/api/turns?limit=${limit}&offset=${offset}`),
  turn: (id: string) => request<Turn>(`/api/turns/${id}`),
  ask: (text: string) =>
    request<Turn>('/api/turns', { method: 'POST', body: JSON.stringify({ text }) }),
  remove: (id: string) => request<void>(`/api/turns/${id}`, { method: 'DELETE' }),
  audioUrl: (id: string) => `/api/audio/${id}`,
}

/** Absolute or relative WebSocket URL, matching the page's own origin. */
export function wsUrl(path = '/ws'): string {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${protocol}//${window.location.host}${path}`
}

export function parseEvent(data: unknown): WsEvent | null {
  if (typeof data !== 'object' || data === null) return null
  const candidate = data as Partial<WsEvent>
  if (
    (candidate.event !== 'turn.created' && candidate.event !== 'turn.updated') ||
    typeof candidate.turn !== 'object' ||
    candidate.turn === null ||
    typeof candidate.turn.id !== 'string'
  ) {
    return null
  }
  return candidate as WsEvent
}
