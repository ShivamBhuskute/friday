# FRIDAY

A wake-word voice assistant. You say **"FRIDAY"**, the ESP32-S3 streams your
sentence over Wi-Fi, and a local pipeline transcribes it, answers it with a
small language model plus tools, and shows the whole exchange in a web console.

## 🎥 Prototype Demo Video

[![Watch the FRIDAY Demo](https://img.youtube.com/vi/DWBUDTLkvM8/maxresdefault.jpg)](https://youtu.be/DWBUDTLkvM8)


Nothing leaves the machine. No cloud STT, no cloud LLM, no API keys.

```
  ESP32-S3                    this machine
  ┌──────────────┐            ┌─────────────────────────────────────┐
  │ "FRIDAY"     │            │  TCP 5000   headerless PCM16        │
  │  heard  ─────┼── Wi-Fi ──▶│      ▼                             │
  │              │  3840-byte  │  utterance framing + VAD             │
  │ one socket   │  hops       │      ▼                             │
  │ per question │            │  faster-whisper  (GPU)              │
  │              │            │      ▼                             │
  │              │            │  Qwen 2.5 3B + tools  (GPU)         │
  │              │            │      ▼                             │
  │              │            │  SQLite  ──▶  web console  :8000    │
  └──────────────┘            └─────────────────────────────────────┘
```

## What is in here

| Path | What it is |
|---|---|
| `friday_kws/` | The ESP32-S3 firmware. Keyword spotting, I2S capture, TCP streaming. |
| `website/` | The PC pipeline and web console. Ingest, STT, LLM agent, tools, UI. |

Start with `website/README.md` for the pipeline in detail. This file is the map
and the test plan.

---

## Quick start (no hardware)

You need Python 3.13, Node, an NVIDIA GPU with CUDA, and about 8 GB of disk.

```bash
cd website
bash scripts/setup.sh          # venv, CUDA wheels, model weights, frontend build
.venv/bin/python scripts/smoke.py
.venv/bin/python -m server
```

Open **<http://localhost:8000>**. That single process serves the API and the
console.

To talk to it without a device, replay a real recording over TCP exactly the
way the firmware sends it:

```bash
.venv/bin/python tools/replay_device.py --quiet fixtures/q_math_7x23.wav
.venv/bin/python tools/replay_device.py --quiet fixtures/q_weather_pune.wav
```

A turn appears in the console with the transcript, the answer, which tools ran,
and a waveform you can play back.

## Quick start (with hardware)

The firmware ships with a placeholder address, so point it at this machine
first:

```bash
cd website && bash scripts/lan_ip.sh
```

That prints the `#define` to paste, reads back what is there now, and says
whether they match. Update it in `friday_kws/main/friday_kws.cpp`, then build
and flash:

```bash
cd friday_kws
. $HOME/esp/esp-idf/export.sh        # or wherever ESP-IDF lives
idf.py set-target esp32s3
idf.py build
idf.py -p /dev/ttyUSB0 flash monitor
```

Both machines must be on the same subnet, and port 5000 must be reachable
(`sudo ufw allow 5000` if a firewall is active). Then say *"FRIDAY, what is
seven times twenty three"*.

The full bench checklist, including how to tell a firmware problem from a
pipeline one, is in [`website/docs/smoke.md`](website/docs/smoke.md).

---

## Testing

Everything below runs without hardware.

### The whole suite

```bash
cd website
.venv/bin/python -m pytest -q      # 476 tests, ~40s
.venv/bin/ruff check .
cd web && npm test && npm run typecheck
```

`pytest` loads the real STT and LLM models, so the run is not instant. The 23
tests that need model weights are marked `slow`; skip them for a fast pass:

```bash
.venv/bin/python -m pytest -q -m "not slow"   # 453 tests, ~25s, no weights
```

### End-to-end checks

```bash
.venv/bin/python scripts/smoke.py
```

Ten checks, each printing the evidence that made it pass — transcript text,
latency, byte counts:

```
pass  imports                      faster-whisper, llama-cpp, numpy
pass  model weights                present (2588 MB)
pass  gpu                          visible to CTranslate2
pass  server health                status=ok stt=lazy llm=ready
pass  frontend served              SPA is mounted at /
pass  speech -> answer (math)      'What is 7 * 23?' -> '161.' (stt 344ms, llm 0ms)
pass  mid-sentence pause is one turn   one turn, 3.58s
pass  speech -> tool call (weather) 'The weather in Pune right now is 28 degrees Celsius...'
pass  audio playback               82520 bytes of valid WAV
pass  typed turn (no hardware)     'what is 7 times 23' -> '161.'
```

### Checking a specific behaviour

```bash
# the wire contract, including a live diff against the firmware's own #defines
.venv/bin/python -m pytest tests/test_protocol.py -v

# replay any clip, with a simulated mid-sentence pause spliced in
.venv/bin/python tools/replay_device.py --gap-ms 900 fixtures/q_weather_pune.wav

# run the agent against a fixed set of questions, no audio involved
.venv/bin/python tools/probe_agent.py
```

### Things to say to FRIDAY

These were run three times each through the HTTP API. All of them reach the
right tool and give the right answer; the weather wording varies a little
between runs because a 3B model is not deterministic in its phrasing.

| Say this | Tool | Typical answer | Time |
| --- | --- | --- | --- |
| What is the weather in Pune right now? | `get_weather` | It is 29.3 degrees in Pune | 1-3 s |
| What is the weather in Mumbai? | `get_weather` | It is 30.4 degrees in Mumbai | 1-2 s |
| Is it raining in London? | `get_weather` | It is not raining in London | 1-2 s |
| What is 7 times 23? | `calculate` | 161 | 0.2 s |
| What is 15 percent of 240? | `calculate` | 36 | 0.2 s |
| What is 2 to the power of 10? | `calculate` | 1024 | 0.2 s |
| What is the current date? | `get_datetime` | The current date is 2026-09-27 | 0.8 s |
| What time is it right now? | `get_datetime` | It is 11:53 in IST | 0.8 s |
| Who are you? | none | I am Friday, how may I assist you today? | 0.6 s |
| Tell me a joke. | none | Why don't scientists trust atoms? | 0.6 s |

The first one is the best thing to demo: it is live data off the network, it
exercises the audio path end to end, and the answer is visibly not something
the model could have guessed.

Avoid "How much memory does this machine have?". The model reaches for
`calculate("memory()")` instead of `system_status` and then invents a number.
That is a tool-selection limit of a 3B model, not a wiring fault, so it is
left as a known limitation rather than papered over.

### Regenerating the speech fixtures

```bash
.venv/bin/python tools/make_fixtures.py     # needs the piper-tts dev extra
```

---

## Configuration

`website/config.yaml` is the single source of truth. Precedence, lowest first:

1. `config.yaml` — committed defaults
2. `config.local.yaml` — your machine, git-ignored
3. `FRIDAY_SECTION__KEY` environment variables

```bash
FRIDAY_INGEST__VAD_RMS_THRESHOLD=0.006 .venv/bin/python -m server
```

Two settings are coupled to the firmware and will bite if you change one
without the other. The server warns at startup when they disagree:

- `ingest.vad_rms_threshold` must stay **below** the device's
  `STREAM_SILENCE_RMS_GATE`, or the server cuts utterances at natural
  mid-sentence pauses and answers one question twice.
- `ingest.max_utterance_s` matches the device's `STREAM_MAX_TIME_MS`.

`website/docs/ingest_contract.md` documents the wire format the PC side is
written against, and the settings that have to agree across the two halves.

---

## Hardware notes

| | |
|---|---|
| Target | ESP32-S3, 4 MB flash, Xtensa |
| Microphone | I2S, 16 kHz mono |
| Toolchain | ESP-IDF, C++17 |
| Build | `idf.py build` — only `main/friday_kws.cpp` is compiled |
| Model | Custom quantised KWS in `friday_kws/models/` |

`friday_kws/deteced_recordings/` holds real captures from the device. They are
the calibration evidence behind `stt.min_confidence`: nine genuine commands
score 0.62–0.83, while false wake-word activations on music and room tone score
0.28–0.45. Replay them through the pipeline to confirm the gates still hold.

## Licence

Private project. No licence granted.
