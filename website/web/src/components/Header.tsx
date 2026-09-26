import type { SocketState } from '../hooks/useTurnFeed'
import type { Health, IngestStats } from '../lib/types'

/**
 * Top bar: is anything actually working?
 *
 * A voice assistant that is silently broken looks identical to one that is idle,
 * so connectivity, the device, and the two models each get their own indicator.
 */
export function Header({
  socket,
  health,
  ingest,
}: {
  socket: SocketState
  health: Health | null
  ingest: IngestStats | null
}) {
  const deviceUp = health?.device_connected ?? false
  const live = socket === 'open'

  return (
    <header className="sticky top-0 z-20 border-b border-ink-800/80 bg-ink-950/80 backdrop-blur-xl">
      <div className="mx-auto flex max-w-6xl flex-wrap items-center gap-x-5 gap-y-2 px-4 py-3 sm:px-6">
        <div className="flex items-baseline gap-2">
          <h1 className="text-[15px] font-semibold tracking-[0.2em] text-ink-100">FRIDAY</h1>
          <span className="font-mono text-[10px] uppercase tracking-wider text-ink-600">console</span>
        </div>

        <div className="flex flex-1 flex-wrap items-center gap-x-4 gap-y-1.5 text-[11px]">
          <Indicator
            label={live ? 'feed live' : socket === 'connecting' ? 'connecting' : 'feed offline'}
            tone={live ? 'ok' : 'warn'}
            title="WebSocket connection to the turn feed"
          />

          <Indicator
            label={
              deviceUp
                ? `device connected${ingest?.port ? ` :${ingest.port}` : ''}`
                : `device idle${ingest?.port ? ` · :${ingest.port}` : ''}`
            }
            tone={deviceUp ? 'ok' : 'idle'}
            title="Whether an ESP32 has connected recently"
          />

          <Indicator
            label={
              health?.stt === 'ready'
                ? `stt ${ingest?.stt_compute_type ?? 'ready'}`
                : `stt ${health?.stt ?? 'down'}`
            }
            tone={health?.stt === 'ready' ? 'ok' : 'warn'}
            title="Speech recognition model"
          />

          <Indicator
            label={`llm ${health?.llm ?? 'down'}`}
            tone={health?.llm === 'ready' ? 'ok' : 'warn'}
            title="Local language model"
          />

          {ingest && ingest.queue_depth > 0 ? (
            <span className="font-mono text-glow-400 tabular-nums">
              {ingest.queue_depth} queued
            </span>
          ) : null}
        </div>

        {ingest?.last_audio_at ? (
          <span className="font-mono text-[10px] text-ink-600 tabular-nums">
            last audio{' '}
            {new Date(ingest.last_audio_at * 1000).toLocaleTimeString([], {
              hour: '2-digit',
              minute: '2-digit',
              second: '2-digit',
              hour12: false,
            })}
          </span>
        ) : null}
      </div>
    </header>
  )
}

function Indicator({
  label,
  tone,
  title,
}: {
  label: string
  /** `warn` pulses, so an unready model is noticed without being alarming. */
  tone: 'ok' | 'warn' | 'idle'
  title: string
}) {
  const colour =
    tone === 'ok'
      ? 'bg-accent-500'
      : tone === 'warn'
        ? 'bg-amber-400 pulse-dot'
        : 'bg-ink-600'
  return (
    <span className="inline-flex items-center gap-1.5 text-ink-400" title={title}>
      <span className={`size-1.5 rounded-full ${colour}`} />
      {label}
    </span>
  )
}
