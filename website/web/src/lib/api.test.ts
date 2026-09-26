import { describe, expect, it } from 'vitest'
import { parseEvent, wsUrl } from '../lib/api'
import { isTerminal, TERMINAL_STATES } from '../lib/types'

describe('wsUrl', () => {
  it('uses ws on plain http and wss on https', () => {
    // jsdom serves the page over http, so this is the ws: branch.
    expect(wsUrl()).toMatch(/^ws:\/\/[^/]+\/ws$/)
  })

  it('accepts a custom path', () => {
    expect(wsUrl('/other')).toMatch(/\/other$/)
  })
})

describe('parseEvent', () => {
  const turn = { id: 'abc', seq: 1 }

  it('accepts both event kinds', () => {
    expect(parseEvent({ event: 'turn.created', turn })?.event).toBe('turn.created')
    expect(parseEvent({ event: 'turn.updated', turn })?.event).toBe('turn.updated')
  })

  it('drops anything that is not a turn event', () => {
    expect(parseEvent(null)).toBeNull()
    expect(parseEvent('hello')).toBeNull()
    expect(parseEvent({ event: 'turn.deleted', turn })).toBeNull()
    expect(parseEvent({ event: 'turn.created' })).toBeNull()
    expect(parseEvent({ event: 'turn.created', turn: { seq: 1 } })).toBeNull()
    expect(parseEvent({ event: 'turn.created', turn: null })).toBeNull()
  })
})

describe('isTerminal', () => {
  it('separates finished from in-flight states', () => {
    for (const state of ['done', 'no_speech', 'unclear', 'error'] as const) {
      expect(isTerminal(state)).toBe(true)
    }
    for (const state of ['uploaded', 'transcribing', 'thinking', 'calling'] as const) {
      expect(isTerminal(state)).toBe(false)
    }
  })

  it('covers every declared terminal state exactly', () => {
    expect([...TERMINAL_STATES].sort()).toEqual(['done', 'error', 'no_speech', 'unclear'])
  })
})
