import { useCallback, useEffect, useState } from 'react'
import { api, ApiError } from '../lib/api'
import type { Health, IngestStats, Stats, SystemStatus, Turn, WsEvent } from '../lib/types'

/**
 * Turn list plus live health.
 *
 * The WebSocket carries every state change, so the list is a straight upsert on
 * each frame -- no optimistic writes, no polling of the list. The side rail
 * numbers are cheap and change independently of turns, so those *are* polled.
 */

const META_INTERVAL_MS = 3000

export interface ConsoleData {
  turns: Turn[]
  health: Health | null
  ingest: IngestStats | null
  stats: Stats | null
  system: SystemStatus | null
  serverError: string | null
  ask: (text: string) => Promise<void>
  remove: (id: string) => Promise<void>
  refresh: () => Promise<void>
  /** Feed handler, exposed so `App` can wire it to the socket. */
  onEvent: (event: WsEvent) => void
  /** Bump to force a re-read of the turn list. */
  resync: () => void
}

/** Newest first, so the answer you just asked for is never below the fold. */
function order(turns: Turn[]): Turn[] {
  return [...turns].sort((a, b) => b.seq - a.seq)
}

function upsert(turns: Turn[], incoming: Turn): Turn[] {
  const existing = turns.findIndex((t) => t.id === incoming.id)
  if (existing === -1) return order([incoming, ...turns])
  const next = [...turns]
  next[existing] = incoming
  return next
}

export function useConsole(): ConsoleData {
  const [turns, setTurns] = useState<Turn[]>([])
  const [health, setHealth] = useState<Health | null>(null)
  const [ingest, setIngest] = useState<IngestStats | null>(null)
  const [stats, setStats] = useState<Stats | null>(null)
  const [system, setSystem] = useState<SystemStatus | null>(null)
  const [serverError, setServerError] = useState<string | null>(null)
  const [reloadToken, setReloadToken] = useState(0)

  const refresh = useCallback(async () => {
    try {
      const [nextTurns, nextHealth, nextIngest, nextStats, nextSystem] = await Promise.all([
        api.turns(50),
        api.health(),
        api.ingestStats(),
        api.stats(),
        api.system().catch(() => null),
      ])
      setTurns(order(nextTurns))
      setHealth(nextHealth)
      setIngest(nextIngest)
      setStats(nextStats)
      if (nextSystem) setSystem(nextSystem)
      setServerError(null)
    } catch (cause) {
      setServerError(
        cause instanceof ApiError ? cause.message : 'cannot reach the FRIDAY server',
      )
    }
  }, [])

  useEffect(() => {
    void refresh()
  }, [refresh, reloadToken])

  // The vitals rail: polled, because nothing pushes these.
  useEffect(() => {
    const timer = setInterval(() => {
      void Promise.all([
        api.health().then(setHealth).catch(() => undefined),
        api.ingestStats().then(setIngest).catch(() => undefined),
        api.stats().then(setStats).catch(() => undefined),
        api.system().then(setSystem).catch(() => undefined),
      ])
    }, META_INTERVAL_MS)
    return () => clearInterval(timer)
  }, [])

  const onEvent = useCallback((event: WsEvent) => {
    setTurns((prev) => upsert(prev, event.turn))
  }, [])

  const ask = useCallback(async (text: string) => {
    const turn = await api.ask(text)
    setTurns((prev) => upsert(prev, turn))
  }, [])

  const remove = useCallback(async (id: string) => {
    setTurns((prev) => prev.filter((t) => t.id !== id))
    try {
      await api.remove(id)
    } catch (cause) {
      // Put it back: a delete that failed must not look like it succeeded.
      setReloadToken((n) => n + 1)
      throw cause
    }
  }, [])

  return {
    turns,
    health,
    ingest,
    stats,
    system,
    serverError,
    ask,
    remove,
    refresh,
    /** Feed handler, exposed so `App` can wire it to the socket. */
    onEvent,
    /** Bump to force a re-read of the turn list. */
    resync: useCallback(() => setReloadToken((n) => n + 1), []),
  }
}
