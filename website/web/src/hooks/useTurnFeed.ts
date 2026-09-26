import { useEffect, useRef, useState } from 'react'
import { parseEvent, wsUrl } from '../lib/api'
import type { WsEvent } from '../lib/types'

export type SocketState = 'connecting' | 'open' | 'closed'

/**
 * Subscribe to the turn feed.
 *
 * The socket carries every state change for every turn, so a client that
 * reconnects after a drop has a hole in its history. The reconnect callback
 * therefore re-fetches the turn list rather than trusting continuity.
 */
export function useTurnFeed(onEvent: (event: WsEvent) => void, onResync?: () => void) {
  const [state, setState] = useState<SocketState>('connecting')

  // Keep the latest callbacks without re-opening the socket on every render.
  const handlerRef = useRef(onEvent)
  const resyncRef = useRef(onResync)
  handlerRef.current = onEvent
  resyncRef.current = onResync

  useEffect(() => {
    let socket: WebSocket | null = null
    let timer: ReturnType<typeof setTimeout> | undefined
    let attempt = 0
    let disposed = false

    const connect = () => {
      if (disposed) return
      setState('connecting')
      try {
        socket = new WebSocket(wsUrl())
      } catch {
        schedule()
        return
      }

      socket.onopen = () => {
        attempt = 0
        setState('open')
      }

      socket.onmessage = (message) => {
        let parsed: unknown
        try {
          parsed = JSON.parse(String(message.data))
        } catch {
          return // a frame we cannot read is not worth crashing the console over
        }
        const event = parseEvent(parsed)
        if (event) handlerRef.current?.(event)
      }

      socket.onclose = () => {
        setState('closed')
        schedule()
      }

      socket.onerror = () => {
        // `onclose` always follows, and that is where reconnection is handled.
        socket?.close()
      }
    }

    const schedule = () => {
      if (disposed) return
      attempt += 1
      // Back off, but never past a few seconds: this is a live console.
      const delay = Math.min(5000, 300 * 2 ** Math.min(attempt, 5))
      timer = setTimeout(() => {
        resyncRef.current?.()
        connect()
      }, delay)
    }

    connect()
    return () => {
      disposed = true
      if (timer) clearTimeout(timer)
      if (socket) {
        socket.onclose = null
        socket.close()
      }
    }
  }, [])

  return state
}
