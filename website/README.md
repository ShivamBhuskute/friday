# FRIDAY

A local voice assistant. Speak to an ESP32-S3, and a few hundred milliseconds
later your laptop has heard you, understood the instruction, run whatever the
request needed, and shown you the answer next to a waveform of what you said.

Nothing leaves the machine. Speech recognition, the language model, and every
tool it can call run on the PC.

```
ESP32-S3                    this machine
────────                    ────────────
  INMP / I2S  ──TCP:5000──▶  framer → WAV
                              │
                              ├─▶ faster-whisper (CUDA) ──▶ "what is the weather in Pune right now?"
                              │
                              ├─▶ Qwen2.5-3B (CUDA) ──▶ get_weather(city="Pune")
                              │                       Open-Meteo ──▶ 28 °C, partly cloudy
                              │                            │
                              └────────────────────────────┴─▶ "It's 28 degrees Celsius in Pune."
                                                          │
                                              web console ──┘
                                     transcript · answer · tools · waveform
```

---

## Quick start

```bash
./scripts/setup.sh          # venv, CUDA llama build, model weights, frontend (~15 min)
.venv/bin/python -m server  # http://localhost:8000
```

No hardware to hand? This replays a recording through the exact protocol the
firmware uses — headerless PCM16 in 3840-byte hops — so the whole pipeline runs:

```bash
.venv/bin/python tools/replay_device.py --file fixtures/q_math_7x23.wav
```

…or just type a question into the console's input box. Typed turns take exactly
the same path as spoken ones.

Is it working? One command, no browser:

```bash
.venv/bin/python scripts/smoke.py
```

---

## What the console gives you

Open <http://localhost:8000>.

- **Every turn, newest first** — the instruction you gave, the answer you got,
  the state it is in right now (`received → transcribing → thinking → calling
  tool → answered`), and per-stage timings.
- **Audio per turn.** A waveform drawn from the actual recording, with
  play/pause, a draggable playhead, and arrow-key seeking. It is your own voice
  in the box, which is how you tell a bad transcription from a bad answer.
- **The tool trace.** Expand it and you see which tools the model called, with
  what arguments, what came back, and how long each took. When an answer is
  wrong, this is where the reason usually is.
- **A vitals rail** — listener status, socket count, bytes received, queue depth,
  average STT and LLM latency, host memory and disk, and the device's own
  sysmon snapshot if it is pushing one.
- **Live updates over a WebSocket.** A new turn appears the moment the audio
  lands, not when the answer is ready.

## Layout

```
server/
  __main__.py      python -m server
  config.py        config.yaml + FRIDAY_SECTION__KEY overrides
  ingest/
    server.py      TCP listener
    stream.py      byte stream -> utterances (WAV framing, raw-PCM VAD)
    wav.py         RIFF parsing, format detection, downmix
  stt.py           faster-whisper, plus transcript normalisation
  agent.py         Qwen tool-calling loop
  tools/           get_weather, calculate, get_datetime, system_status
  pipeline.py      worker thread: audio/text in, turn out
  db.py            SQLite, one row per turn
  bus.py           pub/sub for the WebSocket feed
  web/api.py       REST + /ws + serves the built frontend
web/               Vite + React 19 + TypeScript + Tailwind 4
  src/lib/         types, api client, formatters, peak extraction
  src/hooks/       turn feed (WebSocket), audio player, console state
  src/components/  TurnCard, Waveform, ToolTrace, StatusPill, Sidebar, …
tests/             476 tests
fixtures/          TTS-generated speech, 9 questions with expected transcripts
tools/             replay_device.py, make_fixtures.py, probe_agent.py
scripts/           setup.sh, smoke.py, fetch_models.py, verify_gpu.py
docs/              ingest_contract.md, smoke.md
```

## How a turn works

1. **Device** connects to TCP 5000 when the wake word fires and streams
   headerless 16 kHz mono PCM16 in 3840-byte hops — no RIFF header — then closes
   the socket to end the utterance. See
   [`docs/ingest_contract.md`](docs/ingest_contract.md) — including the
   32-bit-I2S-slot trap that makes "correct" firmware sound garbled, and the
   silence gate the server has to stay looser than.
2. **Framer** segments the byte stream into utterances, detecting a WAV header
   if there is one and falling back to energy-based endpointing if there isn't.
   The device endpointed the audio itself, so the socket close is the signal.
3. **Pipeline** writes a turn row and hands the PCM to a single worker thread.
4. **STT** transcribes on the GPU (~100 ms for a 2-second clip) and normalises
   the result: spoken numbers to digits, `"twenty-three"` to `23 + 7`, wake-word
   and filler words stripped.
5. **Agent** answers. Unambiguous arithmetic (`"what is 7 times 23"`) is
   evaluated deterministically by the `calculate` tool without waking the LLM
   at all, so it answers in ~1 ms. Everything else goes to Qwen2.5-3B with the
   tools attached; if the request needs live data, the model calls
   `get_weather` and the result is folded back into a sentence.
6. **Feed** broadcasts each state change over the WebSocket, and the turn is
   persisted to SQLite.

## Tools

| Tool | What it does |
|---|---|
| `get_weather` | Live conditions via Open-Meteo (keyless, no account). Geocodes the city, then fetches the forecast. |
| `calculate` | An AST-whitelisted expression evaluator. Parses spoken forms — scales, compounds, `"what is"`, hyphens, thousands separators. |
| `get_datetime` | Current local time and date. |
| `system_status` | CPU, memory, disk, plus the device's free heap and Wi-Fi RSSI if it has published a snapshot. |

The calculator never hands the string to `eval`. It tokenises, parses to an AST,
and walks it against a node whitelist, so a transcript cannot execute anything.

## Configuration

Everything lives in [`config.yaml`](config.yaml) — ports, model paths, STT
compute type, decoder hotwords, LLM sampling, tool timeouts, retention.

Anything in it can be overridden per-process:

```bash
FRIDAY_STT__DEVICE=cpu FRIDAY_SERVER__PORT=9000 .venv/bin/python -m server
```

The knobs worth knowing:

| Key | Default | Why you would change it |
|---|---|---|
| `ingest.port` | `5000` | Must match what the firmware dials. |
| `stt.model` | `small.en` | A larger model is more accurate and slower. |
| `stt.hotwords` | a city list | The fix when a proper noun is misheard. |
| `llm.n_gpu_layers` | `-1` | Set `0` to force CPU if the GPU build misbehaves. |
| `llm.chat_format` | `chatml-function-calling` | Do not change: plain `chatml` makes the model ignore the tools and do the arithmetic itself, wrong. |
| `tools.calc_fast_path` | `true` | `false` routes arithmetic through the LLM too. |
| `retention.max_turns` | `200` | Turn history kept; `0` is unlimited. |

## Tests

```bash
.venv/bin/python -m pytest -q     # 476 Python tests, ~40s (models loaded)
.venv/bin/python -m ruff check .
cd web && npm test                # 82 frontend tests, ~1s
```

The Python suite covers the WAV parser, the stream framer, the calculator, the
weather client, the database, the pipeline state machine, the ingest listener
over a real socket, the REST and WebSocket API, and both models end to end.
`tests/test_end_to_end.py` is the one that would catch a broken contract
between the firmware and the server: it streams real speech fixtures over a real
socket and asserts on the answer that comes back.

Tests marked `slow` need the downloaded weights. Skip them with `-m "not slow"`
— the rest runs in about 15 seconds.

## Requirements

- **Python 3.13**, a venv, and roughly 6 GB of disk (2.5 GB of weights, 3 GB of
  dependencies).
- **An NVIDIA GPU is strongly recommended.** Without one, STT falls back to CPU
  int8 and each utterance takes tens of seconds. `scripts/verify_gpu.py` tells
  you what the stack can actually see.
- **Node 20.19+** to build the frontend.
- **CUDA**: the wheels pin `libcublas.so.12` and `libcudnn.so.9` via
  `nvidia-cublas-cu12` / `nvidia-cudnn-cu12`, and `server/cuda_compat.py`
  preloads them. A CUDA 13 toolkit alone does not work, because CTranslate2
  wants the `.so.12` line.

## Hardware

The ESP32-S3 side is a separate codebase. What this end needs from it is
summarised in [`docs/ingest_contract.md`](docs/ingest_contract.md); the bench
checklist is in [`docs/smoke.md`](docs/smoke.md).
