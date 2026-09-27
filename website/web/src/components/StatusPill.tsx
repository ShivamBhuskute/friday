import type { TurnState } from '../lib/types'

/**
 * The pipeline's state, as a small coloured pill.
 *
 * The labels are the pipeline's own vocabulary ("transcribing", "thinking") so
 * that what is on screen matches what the server logs -- if something hangs, the
 * pill says which stage hung.
 */

const STYLES: Record<TurnState, { label: string; className: string }> = {
  uploaded: { label: 'received', className: 'text-ink-300 bg-ink-800 border-ink-700' },
  transcribing: {
    label: 'transcribing',
    className: 'text-accent-400 bg-accent-500/10 border-accent-500/30',
  },
  thinking: {
    label: 'thinking',
    className: 'text-glow-400 bg-glow-500/10 border-glow-500/30',
  },
  calling: {
    label: 'calling tool',
    className: 'text-glow-400 bg-glow-500/10 border-glow-500/30',
  },
  done: { label: 'answered', className: 'text-accent-500 bg-accent-500/10 border-accent-500/25' },
  no_speech: { label: 'no speech', className: 'text-ink-400 bg-ink-800/60 border-ink-700' },
  unclear: { label: 'unclear', className: 'text-amber-300 bg-amber-400/10 border-amber-400/30' },
  error: { label: 'failed', className: 'text-rose-300 bg-rose-500/10 border-rose-500/30' },
}

const BUSY: ReadonlySet<TurnState> = new Set<TurnState>([
  'uploaded',
  'transcribing',
  'thinking',
  'calling',
])

export function isBusy(state: TurnState): boolean {
  return BUSY.has(state)
}

export function statusLabel(state: TurnState): string {
  return STYLES[state]?.label ?? state
}

export function StatusPill({ state }: { state: TurnState }) {
  const style = STYLES[state] ?? STYLES.error!
  const busy = isBusy(state)
  return (
    <span
      data-testid="status-pill"
      data-state={state}
      className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-sm font-medium tracking-wide ${style.className}`}
    >
      {busy ? <span className="pulse-dot size-1.5 rounded-full bg-current" /> : null}
      {style.label}
    </span>
  )
}
