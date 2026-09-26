import { useState } from 'react'
import { formatJson, formatMs, summariseResult } from '../lib/format'
import type { ToolCall } from '../lib/types'

/**
 * What the model actually did.
 *
 * This exists so that a wrong answer can be diagnosed without reading server
 * logs: if the temperature is wrong, `get_weather`'s arguments show the city the
 * model passed, and its result shows what came back.
 */
export function ToolTrace({ calls }: { calls: ToolCall[] }) {
  const [open, setOpen] = useState(false)
  if (calls.length === 0) return null

  const totalMs = calls.reduce((sum, call) => sum + (call.duration_ms ?? 0), 0)
  const failed = calls.some((call) => call.error)

  return (
    <div className="mt-2">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
        className="flex items-center gap-2 text-left text-[11px] text-ink-400 transition hover:text-ink-200"
      >
        <svg
          viewBox="0 0 8 8"
          aria-hidden="true"
          className={`size-2 transition-transform ${open ? 'rotate-90' : ''}`}
        >
          <path d="M2 0l4 4-4 4z" fill="currentColor" />
        </svg>
        <span className="font-mono text-ink-500">
          {calls.length} {calls.length === 1 ? 'tool' : 'tools'}
        </span>
        <span aria-hidden="true">·</span>
        <span className="truncate font-mono text-ink-500">
          {calls.map((call) => call.name).join(' -> ')}
        </span>
        {totalMs > 0 ? (
          <>
            <span aria-hidden="true">·</span>
            <span className="font-mono text-ink-500 tabular-nums">{formatMs(totalMs)}</span>
          </>
        ) : null}
        {failed ? <span className="text-rose-300">· failed</span> : null}
      </button>

      {open ? (
        <ol className="mt-2 space-y-1.5 border-l border-ink-700 pl-3">
          {calls.map((call, index) => (
            <li key={`${call.name}-${index}`} className="font-mono text-[11px]">
              <div className="flex flex-wrap items-baseline gap-x-2">
                <span className="text-glow-400">{call.name}</span>
                <span className="text-ink-500">({compactArgs(call.arguments)})</span>
                {call.duration_ms !== null ? (
                  <span className="text-ink-600 tabular-nums">{formatMs(call.duration_ms)}</span>
                ) : null}
              </div>
              {call.error ? (
                <p className="text-rose-300">{call.error}</p>
              ) : (
                <p className="text-ink-300">
                  <span className="text-ink-600">{'-> '}</span>
                  {summariseResult(call.result)}
                </p>
              )}
            </li>
          ))}
        </ol>
      ) : null}
    </div>
  )
}

/** `{"city": "Pune"}` -> `city="Pune"`, so a scan reads like code, not JSON. */
function compactArgs(args: Record<string, unknown>): string {
  const entries = Object.entries(args ?? {})
  if (entries.length === 0) return ''
  return entries
    .map(([key, value]) => {
      if (value === null || value === undefined) return `${key}=null`
      if (typeof value === 'object') return `${key}=${JSON.stringify(value)}`
      return `${key}=${JSON.stringify(value)}`
    })
    .join(', ')
}

/** Exported for the expanded view in the detail pane. */
export function ToolCallDetail({ call }: { call: ToolCall }) {
  return (
    <pre className="overflow-x-auto rounded-lg bg-ink-900 p-3 font-mono text-[11px] leading-relaxed text-ink-200">
      {formatJson({
        tool: call.name,
        arguments: call.arguments,
        result: call.error ? { error: call.error } : call.result,
      })}
    </pre>
  )
}
