import { formatAgo, formatClock, formatConfidence, formatMs } from '../lib/format'
import { isBusy, StatusPill } from './StatusPill'
import { ToolTrace } from './ToolTrace'
import { Waveform } from './Waveform'
import type { Turn } from '../lib/types'

/**
 * One voice interaction.
 *
 * Two things have to be readable without clicking: what was asked, and what came
 * back. The waveform and the tool trace are secondary, which is why they sit
 * below a rule rather than competing with the two text columns.
 *
 * `latest` marks the turn that just arrived. It is the one the room is looking
 * at, so it gets a bigger instruction, a brighter answer and an accent border,
 * while older turns step back to ordinary body size. Without that, a screen
 * full of history makes the newest exchange no easier to pick out than any
 * other, which is the opposite of what a live console is for.
 */
export function TurnCard({
  turn,
  onDelete,
  latest = false,
}: {
  turn: Turn
  onDelete: (id: string) => void
  latest?: boolean
}) {
  const busy = isBusy(turn.state)
  const confidence = formatConfidence(turn.confidence)
  const spoken = Boolean(turn.audio_url)

  return (
    <article
      data-testid="turn-card"
      data-state={turn.state}
      data-latest={latest ? 'true' : 'false'}
      className={`group relative overflow-hidden rounded-xl border transition-colors ${
        busy
          ? 'sweep border-accent-500/30 bg-ink-850'
          : turn.state === 'error'
            ? 'border-rose-500/25'
            : latest
              ? 'border-accent-500/40 bg-ink-850/80 shadow-lg shadow-accent-500/5'
              : 'border-ink-800 bg-ink-900/60 hover:border-ink-700'
      }`}
    >
      <div className={`flex items-start gap-4 ${latest ? 'p-5' : 'p-4'}`}>
        {/* Left rail: when it happened and how long the audio was. */}
        <div className="hidden w-16 shrink-0 flex-col items-end gap-1 pt-1 sm:flex">
          <time
            dateTime={turn.created_at}
            title={new Date(turn.created_at).toLocaleString()}
            className={`font-mono tabular-nums ${
              latest ? 'text-sm text-ink-300' : 'text-sm text-ink-400'
            }`}
          >
            {formatClock(turn.created_at)}
          </time>
          <span className="font-mono text-xs text-ink-600 tabular-nums">#{turn.seq}</span>
        </div>

        <div className="min-w-0 flex-1">
          <div className={`flex flex-wrap items-center gap-2 ${latest ? 'mb-3' : 'mb-2'}`}>
            <StatusPill state={turn.state} />
            {latest ? (
              <span className="font-mono text-xs font-medium uppercase tracking-wider text-accent-500">
                latest
              </span>
            ) : null}
            {turn.source ? (
              <span className="font-mono text-xs uppercase tracking-wider text-ink-600">
                {turn.source}
              </span>
            ) : null}
            {confidence ? (
              <span
                title="Speech recognition confidence"
                className="font-mono text-xs text-ink-500 tabular-nums"
              >
                {confidence}
              </span>
            ) : null}
            <span className="ml-auto font-mono text-xs text-ink-600">
              {formatAgo(turn.created_at)}
            </span>
            <button
              type="button"
              onClick={() => onDelete(turn.id)}
              aria-label="Delete this turn"
              className="text-ink-600 opacity-0 transition group-hover:opacity-100 hover:text-rose-300 focus-visible:opacity-100"
            >
              <svg viewBox="0 0 12 12" className="size-3.5" aria-hidden="true">
                <path
                  d="M2.5 2.5l7 7M9.5 2.5l-7 7"
                  stroke="currentColor"
                  strokeWidth="1.4"
                  strokeLinecap="round"
                />
              </svg>
            </button>
          </div>

          {/* The instruction. Largest thing on the card when it just arrived. */}
          <p
            data-testid="turn-instruction"
            className={`leading-snug ${
              latest
                ? 'text-2xl font-medium tracking-tight text-ink-100'
                : 'text-base text-ink-200'
            }`}
          >
            {turn.transcript ? (
              <>
                <span
                  aria-hidden="true"
                  className={`mr-2 select-none ${latest ? 'text-accent-500' : 'text-ink-600'}`}
                >
                  &rsaquo;
                </span>
                {turn.transcript}
              </>
            ) : (
              <span className="italic text-ink-600">{busy ? 'listening…' : 'no transcript'}</span>
            )}
          </p>

          {/* The answer. */}
          {turn.answer ? (
            <p
              data-testid="turn-answer"
              className={`mt-2 leading-snug ${
                latest ? 'text-xl text-accent-400' : 'text-base text-accent-500/80'
              }`}
            >
              {turn.answer}
            </p>
          ) : turn.error ? (
            <p data-testid="turn-error" className="mt-2 text-base text-rose-300">
              {turn.error}
            </p>
          ) : null}

          {turn.tool_calls.length > 0 ? <ToolTrace calls={turn.tool_calls} /> : null}
        </div>
      </div>

      {spoken ? (
        <div className="border-t border-ink-800/80 px-4 py-2.5">
          <Waveform url={turn.audio_url} durationS={turn.duration_s} />
        </div>
      ) : null}

      <div className="flex items-center gap-3 border-t border-ink-800/80 px-4 py-2 font-mono text-xs text-ink-600 tabular-nums">
        <span title="Time spent transcribing">stt {formatMs(turn.transcript_ms)}</span>
        <span title="Time spent in the agent, including tool calls">
          llm {formatMs(turn.llm_ms)}
        </span>
        {turn.duration_s !== null ? <span>audio {turn.duration_s.toFixed(2)}s</span> : null}
        {turn.sample_rate ? (
          <span>
            {turn.sample_rate / 1000}kHz {turn.channels === 1 ? 'mono' : 'st'}
          </span>
        ) : null}
        <span className="ml-auto truncate text-ink-700">{turn.id.slice(0, 8)}</span>
      </div>
    </article>
  )
}
