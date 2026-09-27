import { useState } from 'react'

/**
 * Type a question instead of saying one.
 *
 * The point is testability without hardware: the whole pipeline (fast path, LLM,
 * tools, persistence, the WebSocket feed) runs identically for typed text, so the
 * console can be developed and demoed with no ESP32 attached.
 */
export function Composer({ onAsk, disabled }: { onAsk: (text: string) => void; disabled?: boolean }) {
  const [text, setText] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const submit = async (event: React.FormEvent) => {
    event.preventDefault()
    const question = text.trim()
    if (!question || busy) return
    setBusy(true)
    setError(null)
    try {
      await onAsk(question)
      setText('')
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'could not send that')
    } finally {
      setBusy(false)
    }
  }

  return (
    <form onSubmit={submit} className="flex flex-col gap-1.5">
      <div className="flex items-center gap-2 rounded-xl border border-ink-800 bg-ink-900/80 px-3 py-2 transition-colors focus-within:border-accent-500/40">
        <span aria-hidden="true" className="font-mono text-base text-ink-600">
          &rsaquo;
        </span>
        <input
          value={text}
          onChange={(event) => setText(event.target.value)}
          placeholder="ask FRIDAY something…"
          aria-label="Ask FRIDAY a question"
          disabled={disabled}
          maxLength={2000}
          className="min-w-0 flex-1 bg-transparent text-base text-ink-100 placeholder:text-ink-600 focus:outline-none disabled:opacity-50"
        />
        <button
          type="submit"
          disabled={disabled || busy || text.trim().length === 0}
          className="shrink-0 rounded-lg border border-accent-500/30 bg-accent-500/10 px-3 py-1 text-sm font-medium text-accent-400 transition hover:bg-accent-500/20 disabled:cursor-not-allowed disabled:opacity-40"
        >
          {busy ? 'sending' : 'send'}
        </button>
      </div>
      {error ? (
        <p role="alert" className="pl-1 text-sm text-rose-300">
          {error}
        </p>
      ) : null}
    </form>
  )
}
