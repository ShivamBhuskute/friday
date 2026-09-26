import { formatBytes } from '../lib/format'
import type { Health, IngestStats, Stats, SystemStatus } from '../lib/types'

/**
 * The right-hand vitals rail.
 *
 * Everything here answers one of two questions: is it working, and how fast.
 * Latency averages come from `/api/stats`, which is computed over the same rows
 * the feed shows, so a number on screen always matches a card above it.
 */
export function Sidebar({
  health,
  ingest,
  stats,
  system,
}: {
  health: Health | null
  ingest: IngestStats | null
  stats: Stats | null
  system: SystemStatus | null
}) {
  const host = system?.host
  const memPct =
    host?.mem_total_gb && host.mem_used_gb !== undefined && host.mem_used_gb !== null
      ? (host.mem_used_gb / host.mem_total_gb) * 100
      : null
  const diskPct =
    host?.disk_total_gb && host.disk_free_gb !== undefined && host.disk_free_gb !== null
      ? (host.disk_free_gb / host.disk_total_gb) * 100
      : null

  return (
    <aside className="flex flex-col gap-5 lg:sticky lg:top-20 lg:self-start">
      <Section title="pipeline">
        <Row label="transitions" value={stats ? String(stats.done) : '--'} />
        <Row label="failures" value={stats ? String(stats.errors) : '--'} tone={stats?.errors ? 'bad' : undefined} />
        <Row label="stored" value={health ? String(health.turns) : '--'} />
        <Row label="queued" value={ingest ? String(ingest.queue_depth) : '--'} />
        <Row
          label="avg stt"
          value={stats?.avg_transcript_ms !== null && stats ? `${stats.avg_transcript_ms}ms` : '--'}
        />
        <Row
          label="avg llm"
          value={stats?.avg_llm_ms !== null && stats ? `${stats.avg_llm_ms}ms` : '--'}
        />
      </Section>

      <Section title="ingest">
        <Row
          label="listener"
          value={ingest ? (ingest.listening ? `port ${ingest.port}` : 'down') : '--'}
          tone={ingest?.listening ? 'good' : 'bad'}
        />
        <Row label="open sockets" value={ingest ? String(ingest.connections) : '--'} />
        <Row label="utterances" value={ingest ? String(ingest.utterances_received) : '--'} />
        <Row label="received" value={ingest ? formatBytes(ingest.bytes_received) : '--'} />
        <Row label="compute" value={ingest?.stt_compute_type ?? '--'} />
        <Row label="device" value={ingest?.stt_device ?? '--'} />
      </Section>

      <Section title="host">
        <Row
          label="load"
          value={host?.load_1m !== null && host?.load_1m !== undefined ? host.load_1m.toFixed(2) : '--'}
        />
        {host?.mem_total_gb ? (
          <Meter
            label="memory"
            used={host.mem_used_gb ?? 0}
            total={host.mem_total_gb}
            pct={memPct}
          />
        ) : (
          <Row label="memory" value="--" />
        )}
        {host?.disk_total_gb ? (
          <Meter
            label="disk"
            used={(host.disk_total_gb ?? 0) - (host.disk_free_gb ?? 0)}
            total={host.disk_total_gb}
            pct={diskPct}
          />
        ) : (
          <Row label="disk" value="--" />
        )}
      </Section>

      {system?.device ? (
        <Section title="device">
          {Object.entries(system.device).map(([key, value]) => (
            <Row
              key={key}
              label={key.replace(/_/g, ' ')}
              value={typeof value === 'number' ? String(value) : String(value ?? '--')}
            />
          ))}
        </Section>
      ) : system?.device_note ? (
        <Section title="device">
          <p className="text-[11px] leading-relaxed text-ink-600">{system.device_note}</p>
        </Section>
      ) : null}
    </aside>
  )
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="rounded-xl border border-ink-800 bg-ink-900/40 p-3.5">
      <h2 className="mb-2.5 font-mono text-[10px] uppercase tracking-[0.18em] text-ink-600">
        {title}
      </h2>
      <dl className="space-y-1.5">{children}</dl>
    </section>
  )
}

function Row({
  label,
  value,
  tone,
}: {
  label: string
  value: string
  tone?: 'good' | 'bad'
}) {
  return (
    <div className="flex items-baseline justify-between gap-3 text-[11px]">
      <dt className="text-ink-500">{label}</dt>
      <dd
        className={`truncate font-mono tabular-nums ${
          tone === 'bad' ? 'text-rose-300' : tone === 'good' ? 'text-accent-400' : 'text-ink-300'
        }`}
      >
        {value}
      </dd>
    </div>
  )
}

function Meter({
  label,
  used,
  total,
  pct,
}: {
  label: string
  used: number
  total: number
  pct: number | null
}) {
  const clamped = pct === null ? 0 : Math.max(0, Math.min(100, pct))
  return (
    <div className="text-[11px]">
      <div className="flex items-baseline justify-between gap-3">
        <dt className="text-ink-500">{label}</dt>
        <dd className="font-mono text-ink-300 tabular-nums">
          {used.toFixed(1)} / {total.toFixed(1)} GB
        </dd>
      </div>
      <div className="mt-1 h-1 overflow-hidden rounded-full bg-ink-800">
        <div
          className={`h-full rounded-full ${clamped > 90 ? 'bg-rose-400' : 'bg-accent-600'}`}
          style={{ width: `${clamped}%` }}
        />
      </div>
    </div>
  )
}
