# Ingest contract

What the ESP32-S3 puts on the wire, and what the PC does with it.

This describes the **shipped** firmware (`friday_kws/main/friday_kws.cpp`). The
numbers below are not aspirations — they are that file's `#define`s, mirrored in
`server/ingest/protocol.py`, which the ingest server and the replay tool both
import so the two sides cannot drift apart.

> The PC side is deliberately tolerant. It sniffs the stream, repairs a bad
> length field, and still accepts a RIFF/WAVE if one turns up. That tolerance is
> a safety net, not the contract — if the firmware and the server disagree, the
> symptom is a half-transcribed sentence rather than an error, so the server
> logs a warning at startup when its config has drifted from these constants.

---

## 1. The short version

```
TCP connect  ->  headerless PCM16 LE, in 3840-byte hops  ->  close()
```

- **Transport:** plain TCP to the PC, port `5000` by default.
- **Format:** 16 000 Hz, mono, signed 16-bit little-endian PCM.
- **Framing:** none. **There is no RIFF/WAVE header and no length prefix.**
- **Chunking:** 1920 samples = 3840 bytes = 120 ms per hop.
- **One connection per utterance**, opened when the wake word fires.
- **End of utterance:** the device stops sending and closes the socket.
- **No handshake, no acknowledgements.** The server never writes back.

Two consequences worth being blunt about:

1. The server has to guess the sample format, because nothing on the wire says
   it. It uses `ingest.sample_rate` / `channels` / `bits` from `config.yaml`, so
   those must match this document. `startup` warns if they do not.
2. Because there is no header, the first bytes of audio are *data*. That is
   fine — `RIFF` in the first four bytes would be a 1-in-2^48 coincidence, and
   the framer also requires `WAVE` at offset 8 before believing it.

---

## 2. Byte layout

There is no header. The connection carries exactly:

| Field | Value |
|-------|-------|
| sample rate | `16000` Hz |
| channels | `1` |
| bits per sample | `16`, signed, little-endian |
| hop size | `1920` samples = `3840` bytes, every 120 ms |

16 kHz × 1 ch × 2 byte = **32 000 bytes per second**. Samples are full-scale
`±32767`, and there is no framing, padding or alignment between hops.

---

## 3. How the device decides an utterance has ended

The device endpointed the audio itself, before it ever looked at the network.
This is the part that has to stay consistent with the server, so it is worth
stating exactly:

| Firmware `#define` | Value | Meaning |
|--------------------|-------|---------|
| `STREAM_SILENCE_RMS_GATE` | `0.007` | a hop quieter than this counts as silence |
| `STREAM_SILENCE_MS` | `800` | this much continuous silence ends the utterance |
| `STREAM_MAX_TIME_MS` | `10000` | hard cap; the utterance is cut here regardless |
| `HOP_MS` | `120` | the resolution of both timers |

`INPUT_GAIN` is `1.0`, so the device's RMS is already normalised to `0..1` —
**the same units the server measures in**. A silence gate of `0.007` here and
`0.007` on the PC mean the same thing.

The device also sends **no pre-roll**. Streaming starts on the hop *after* the
one the wake word was detected in, so roughly the last 120 ms of "Friday" is
discarded and the recording begins at the start of your actual command. Do not
expect leading audio, and do not go looking for a ring buffer that captures
before the trigger — there isn't one on this path.

### The rule that follows from this

**The server's VAD must be looser than the device's gate.**

The device has already decided where the utterance ends, and the socket close
confirms it. The server's energy VAD is a safety net for a device that dies
mid-sentence, so it should essentially never fire first. Hence:

```yaml
ingest:
  vad_silence_s: 0.8        # == the device's STREAM_SILENCE_MS
  vad_rms_threshold: 0.006  # <  the device's STREAM_SILENCE_RMS_GATE (0.007)
  max_utterance_s: 10.0     # == the device's STREAM_MAX_TIME_MS
```

If `vad_rms_threshold` sits at or above `0.007`, the server concludes the
speaker stopped at the first pause it cannot hear over, and cuts the
utterance in half. On "what is the weather … in Pune", with the beat in the
middle, that is **two turns and two answers to one question** — the first half
gets a confident wrong answer, the second half gets another. This is not
hypothetical; it is what a `0.02` threshold does, and it is why the server
warns at startup rather than quietly misbehaving.

The server's own gate can then only fire when the device has stopped sending
anyway, which makes the effective end-of-utterance the socket close.

### Two device-side limits that cost accuracy

These are properties of the current firmware, not something the PC can fix.
Both are worth knowing before you spend an afternoon blaming the STT model.

**1. There is no pre-roll, so the first syllable can be clipped.** The hop the
wake word is detected in is not streamed — streaming starts on the hop after
it — so up to 120 ms is lost at the front. If you say "FRIDAY what's seven
times twenty three" with no pause, the start of "what's" can be cut. It is
usually survivable, because what is lost is the tail of the wake word itself
and Whisper's VAD discards that, but speaking a beat after the wake word is
more reliable than speaking over it. The fix, if it ever matters, is a ring
buffer in the capture task holding ~2 s and flushed on `STREAM_MSG_START`.

**2. Hops are dropped when the queue is full.** `stream_queue` is
`xQueueCreate(4, ...)` and the capture task uses `xQueueSend(..., 0)` — no
wait. Four hops is 480 ms of audio, so if `send()` blocks for half a second
(Wi-Fi stall, TCP backpressure) the queue fills and further hops are discarded
with a `Stream: queue full, dropping hop` line in the serial log. The result is
a hole in the middle of the sentence.

This one is **not recoverable on the PC**, and that is the important part:
raw PCM16 has no sequence numbers, so a dropped hop and a silence are
indistinguishable by the time they reach the server. If you see a transcript
that is correct but missing a word in the middle, look for that log line.
Buffering more hops, or blocking on the send, would close the gap.

**3. The wake word fires on media playing in the room.** Every recording in
`deteced_recordings/` that produced text was a false activation: a speaker
playing music, a video outro, room tone. None of them was a command, and the
worst ran to the `STREAM_MAX_TIME_MS` cap at 10.08 s of near-silence.

The server absorbs most of this — `stt.min_confidence` and `is_repetitive()`
reject them before the LLM sees them — but they still cost a turn card and a
transcription each. If a demo is going to run next to a speaker, the KWS side is
where the fix belongs: a higher `HITS_REQUIRED`, or refusing to stream while
the detected audio is steady enough to be music rather than speech.

---

## 4. Connection lifecycle

```
ESP32                                        PC
  |                                            |
  |  (wake word "FRIDAY" detected)              |
  |------- TCP connect to <pc-ip>:5000 ------->|
  |                                            |  (no reply expected)
  |------- 3840 bytes (hop 1, 120 ms) -------->|
  |------- 3840 bytes (hop 2) ---------------->|
  |------- ...                                ->|
  |------- 3840 bytes (hop n) ---------------->|
  |       (800 ms of silence, or 10 s elapsed)  |
  |------- close() --------------------------->|  end of utterance
  |                                            |  STT + LLM run
  |                                            |
  |  (next utterance: a new connection)         |
```

- **One utterance per connection.** A connection carries exactly one utterance
  and carries it completely. Reconnect for the next one.
- **Chunk size is irrelevant to the server**, but the device sends 3840 bytes
  per hop, and `tools/replay_device.py` reproduces that, so the reassembly path
  gets exercised.
- **The socket close is the signal.** There is no terminator packet.
- **No reply, ever.** Do not wait for an acknowledgement and do not treat
  silence as a failure.

---

## 5. Two settings on the device that must match the PC

These are the only two things that have to be agreed out of band, and both are
compile-time.

| Firmware `#define` | Must be |
|--------------------|---------|
| `STREAM_SERVER_IP` | the PC's LAN address **on the network the ESP32 joined** |
| `STREAM_SERVER_PORT` | `5000`, matching `ingest.port` |

`STREAM_SERVER_IP` is currently a placeholder and it is wrong for most
networks. Get the right value with:

```bash
scripts/lan_ip.sh
```

The PC and the ESP32 must be on the **same subnet** — the firmware has no
mDNS, no discovery and no fallback, so a PC on a different network is simply
unreachable and the only symptom on the device is
`Stream: connect() failed, errno ...` in the serial log.

Also confirm the PC is not blocking the port:

- `server.host` / `ingest.host` must be `0.0.0.0`, not `127.0.0.1`.
- On Linux, `sudo ufw allow 5000/tcp`. On Windows, the firewall will prompt on
  first bind — allow it on the **private** profile, not public.

---

## 6. The 16-bit vs 32-bit I2S trap — how to tell if it bit you

The ESP32-S3 I2S peripheral shifts out a fixed-width sample per slot, and the
common configurations give you **32 bits per slot, left-justified within it**.
The firmware reads `int32_t` out of the DMA ring and narrows it with:

```c
int16_t raw_smp = (int16_t)(i2s_raw[i] >> 16);   // friday_kws.cpp
```

**That shift is correct** and is what makes the stream 16-bit. If it is ever
removed, or if the peripheral is reconfigured to 16-bit slots while the shift
stays, the audio is wrong in a way that is easy to miss because the pipeline
still produces *a* transcript:

1. **Plays 4× too fast.** Labelling 16 kHz data that is really 4× oversampled
   squeezes a 2-second utterance into 0.5 seconds of timeline.
2. **Wrong amplitude**, by a factor of 256, because you keep the low half of a
   left-justified 32-bit word (mostly zeros) instead of the high half.

Symptoms, in order of how quickly you would notice them:

| What you see | What it means |
|--------------|---------------|
| Transcript is a garbled fragment of the real sentence | 4× rate mismatch, i.e. the `>> 16` is missing |
| Transcript is right but the waveform is a flat line near zero | amplitude collapsed, i.e. the low half is being kept |
| Every clip is ~4× shorter than you spoke | rate mismatch, measurable without the STT at all |

The last row is the cheapest check and needs no models: time yourself speaking
and compare against the reported clip duration. The server logs
`utterance N from #M: <duration>s ... via <source>` for every turn.

---

## 7. What the server tolerates

These all work today. None of them are things to rely on.

| You send | What happens |
|----------|--------------|
| Headerless raw PCM16 | **The primary path.** End-of-utterance is the socket close, backed by a loose energy VAD. |
| A RIFF/WAVE file | Parsed, including a wrong or placeholder `data` size, which is repaired from what arrived. |
| `WAVE_FORMAT_EXTENSIBLE` (0xFFFE) | Parsed via the sub-format GUID. |
| Extra chunks (`LIST`, `fact`) | Skipped by walking the chunk sizes. |
| A `fmt ` chunk of 18 or 40 bytes | Handled; the extra bytes are ignored. |
| 8/24/32-bit or stereo PCM | Downmixed and narrowed to 16 kHz mono. Endpointed on size, since there is no decoder to VAD with. |
| Several connections at once | Up to `ingest.max_connections` (4) are served. |

Because the format is inferred from config and not from the wire, a device that
is *not* 16 kHz mono 16-bit will be transcribed as nonsense rather than
rejected — hence the startup check.

---

## 8. Timing, from the firmware's side

The server has no real-time requirement — it processes each utterance as it
arrives. The constraints that do exist:

- **Utterance length:** the device stops at `10 s`, and the server drops
  anything under `0.3 s` as noise. See `ingest.min_utterance_s` /
  `max_utterance_s`.
- **Idle timeout:** a connection is held open for `300 s` before being closed.
  The device never holds one open that long, so this only reaps abandoned
  sockets.
- **Latency budget on the PC:** STT ~100–360 ms, agent ~0 ms for arithmetic via
  the fast path and ~1–8 s when a tool is called (the weather tool makes two
  HTTP calls). The UI shows progress, so a slow turn is visible as
  `thinking` / `calling tool` rather than looking hung.

---

## 9. Verifying it

### Without hardware

```bash
# 1. Start the server.
.venv/bin/python -m server

# 2. Replay a recorded question through the real protocol.
.venv/bin/python tools/replay_device.py fixtures/q_math_7x23.wav

# 3. Read back what the pipeline made of it.
curl -s localhost:8000/api/turns | python3 -m json.tool
```

`tools/replay_device.py` sends headerless PCM in 3840-byte hops and stops at
the device's own silence gate, so the framing is identical to the firmware's.
To exercise the mid-sentence-pause case from §3 explicitly:

```bash
.venv/bin/python tools/replay_device.py --gap-ms 900 fixtures/q_weather_pune.wav
```

That must produce **one** turn, not two.

### With hardware

`docs/smoke.md` is the bench checklist, with one piece of evidence to collect
per step. The two steps that catch real problems first are: confirm the device
logged `Stream: connected to <ip>:5000`, and confirm the turn's reported
duration matches roughly how long you spoke.

---

## 10. Where the server-side rules live

| Behaviour | File |
|-----------|------|
| **The firmware's constants, mirrored** | `server/ingest/protocol.py` |
| Header parsing, format detection, downmix | `server/ingest/wav.py` |
| Stream → utterance framing, raw VAD | `server/ingest/stream.py` |
| TCP listener, connection lifecycle | `server/ingest/server.py` |
| Startup drift check against the firmware | `server/ingest/server.py` → `check_alignment` |
| Tuning knobs | `config.yaml` → `ingest:` |
| Faithful replay of the device protocol | `tools/replay_device.py` |
| Tests that pin every rule above | `tests/test_protocol.py`, `tests/test_ingest.py`, `tests/test_wav.py` |
