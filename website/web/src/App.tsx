import { Composer } from './components/Composer'
import { Header } from './components/Header'
import { Sidebar } from './components/Sidebar'
import { TurnCard } from './components/TurnCard'
import { useConsole } from './hooks/useConsole'
import { useTurnFeed } from './hooks/useTurnFeed'

export default function App() {
  const { turns, health, ingest, stats, system, serverError, ask, remove, onEvent, resync } =
    useConsole()
  const socket = useTurnFeed(onEvent, resync)

  return (
    <div className="min-h-screen">
      <Header socket={socket} health={health} ingest={ingest} />

      <main className="mx-auto grid max-w-6xl gap-6 px-4 py-6 sm:px-6 lg:grid-cols-[1fr_240px]">
        <div className="min-w-0">
          {serverError ? (
            <div
              role="alert"
              className="mb-4 flex items-center justify-between gap-3 rounded-xl border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-[13px] text-rose-200"
            >
              <span>
                <span className="font-medium">offline.</span> {serverError}
              </span>
              <button
                type="button"
                onClick={() => void resync()}
                className="shrink-0 rounded-lg border border-rose-400/40 px-2.5 py-1 text-[11px] transition hover:bg-rose-400/10"
              >
                retry
              </button>
            </div>
          ) : null}

          <div className="mb-4">
            <Composer onAsk={ask} disabled={serverError !== null} />
          </div>

          {turns.length === 0 ? (
            <EmptyState socket={socket} />
          ) : (
            <ol className="space-y-2.5" data-testid="turn-list">
              {turns.map((turn) => (
                <li key={turn.id}>
                  <TurnCard turn={turn} onDelete={(id) => void remove(id)} />
                </li>
              ))}
            </ol>
          )}
        </div>

        <Sidebar health={health} ingest={ingest} stats={stats} system={system} />
      </main>
    </div>
  )
}

function EmptyState({ socket }: { socket: 'connecting' | 'open' | 'closed' }) {
  return (
    <div className="rounded-xl border border-dashed border-ink-800 px-6 py-14 text-center">
      <p className="text-[13px] text-ink-400">
        {socket === 'open'
          ? 'No turns yet.'
          : socket === 'connecting'
            ? 'Connecting to the FRIDAY server…'
            : 'Disconnected from the FRIDAY server.'}
      </p>
      <p className="mx-auto mt-3 max-w-md text-[12px] leading-relaxed text-ink-600">
        Say something after the wake word, or type a question above. To replay a recorded
        question without hardware:{' '}
        <code className="font-mono text-ink-500">python tools/replay_device.py</code>
      </p>
    </div>
  )
}
