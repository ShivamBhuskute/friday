/** Small formatters. Kept pure and tested because they drive the whole layout. */

/** "1.4s" / "0.32s" / "3ms" — short enough to sit in a table column. */
export function formatDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return '--'
  if (seconds < 1) return `${Math.round(seconds * 1000)}ms`
  if (seconds < 60) return `${seconds.toFixed(seconds < 10 ? 2 : 1)}s`
  const mins = Math.floor(seconds / 60)
  return `${mins}m ${Math.round(seconds % 60)}s`
}

/** Latency in milliseconds, or "--" when the stage never ran. */
export function formatMs(ms: number | null | undefined): string {
  if (ms === null || ms === undefined || !Number.isFinite(ms)) return '--'
  if (ms < 1) return '<1ms'
  if (ms < 1000) return `${Math.round(ms)}ms`
  return `${(ms / 1000).toFixed(2)}s`
}

/** Wall-clock time of an ISO timestamp, in the viewer's locale. */
export function formatClock(iso: string | null | undefined): string {
  if (!iso) return '--'
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return '--'
  return date.toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  })
}

/** Whole percentage, or null when the model gave no confidence. */
export function formatConfidence(value: number | null | undefined): string | null {
  if (value === null || value === undefined || !Number.isFinite(value)) return null
  return `${Math.round(Math.max(0, Math.min(1, value)) * 100)}%`
}

/** Byte counts, for the ingest counter. */
export function formatBytes(bytes: number | null | undefined): string {
  if (!bytes) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB']
  let value = bytes
  let unit = 0
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024
    unit += 1
  }
  return `${value < 10 && unit > 0 ? value.toFixed(1) : Math.round(value)} ${units[unit]}`
}

/** Relative age, e.g. "just now", "4m ago". */
export function formatAgo(iso: string | null | undefined, now = Date.now()): string {
  if (!iso) return '--'
  const then = new Date(iso).getTime()
  if (Number.isNaN(then)) return '--'
  const seconds = Math.max(0, Math.round((now - then) / 1000))
  if (seconds < 5) return 'just now'
  if (seconds < 60) return `${seconds}s ago`
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`
  return `${Math.floor(seconds / 86400)}d ago`
}

/** One-line summary of a tool result, for the collapsed trace row. */
export function summariseResult(value: unknown): string {
  if (value === null || value === undefined) return 'no result'
  if (typeof value === 'string') return value
  if (typeof value === 'number' || typeof value === 'boolean') return String(value)
  if (typeof value === 'object') {
    const record = value as Record<string, unknown>
    // The interesting field is named differently per tool; try the likely ones.
    for (const key of ['result', 'summary', 'answer', 'text', 'expression', 'timezone', 'now']) {
      const found = record[key]
      if (typeof found === 'string' || typeof found === 'number') return String(found)
    }
    const keys = Object.keys(record)
    return keys.length ? `{${keys.slice(0, 4).join(', ')}}` : '{}'
  }
  return String(value)
}

/** Compact JSON for the expanded trace view. */
export function formatJson(value: unknown): string {
  try {
    return JSON.stringify(value, null, 2) ?? String(value)
  } catch {
    return String(value)
  }
}
