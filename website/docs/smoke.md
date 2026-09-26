# Hardware smoke test

Run through this on the bench with the ESP32-S3 plugged in, before the demo.
Each step names the *evidence* that it passed — a state to look for, a number
to read — so "seems fine" doesn't get mistaken for working.

Automated equivalent: `.venv/bin/python scripts/smoke.py` covers steps 0–4 with
no hardware. Steps 5+ need the real device.

---

## 0. The PC is ready

```bash
.venv/bin/python scripts/smoke.py
```

All checks green. If `gpu` says "not visible", STT is on CPU and every utterance
will take tens of seconds — fix that before continuing.

**Evidence:** exit code 0, last line reads `all checks passed`.

---

## 1. The listener is up and reachable

```bash
.venv/bin/python -m server
curl -s localhost:8000/api/health
```

```json
{"status":"ok","ingest":"listening","stt":"ready","llm":"ready","device_connected":false,"turns":0}
```

- `ingest` must be `"listening"`. If `"down"`, the port is taken — something
  else is on 5000, or a previous run is still alive (`pkill -f "python -m server"`).
- `device_connected: false` is correct at this point. Nothing has connected yet.

**Evidence:** the log line `ingest listening on 0.0.0.0:5000`, and `"ingest":"listening"`.

Also worth checking, since it is free: the server logs a warning at startup if
its config has drifted from the firmware's own constants. There should be no
`ingest config does not match the firmware` line.

---

## 1b. The device is pointed at this PC

`STREAM_SERVER_IP` in the firmware is a hard-coded placeholder, and it is
wrong for most networks:

```bash
scripts/lan_ip.sh
```

That prints the address to paste in, reads back what the firmware currently
has, and tells you whether they match. Update the `#define`, rebuild, reflash.
See `docs/ingest_contract.md` §5 for the firewall and same-subnet requirements.

**Evidence:** `scripts/lan_ip.sh` ends with `Matches. No change needed.`

---

## 2. Network reachability from the device's laptop

From the machine that will run the firmware:

```bash
nc -vz <pc-ip> 5000
```

Also confirm the PC is not behind a firewall that drops it:

```bash
sudo ufw status          # if it is active, allow 5000 and 8000
```

**Evidence:** `succeeded`, from the *other* machine, not localhost.

---

## 3. A replayed recording produces a full turn

With the server still running:

```bash
.venv/bin/python tools/replay_device.py --file fixtures/q_math_7x23.wav
```

Then open <http://localhost:8000> and check all of:

- A card appears within a second or two, labelled `received`, then `transcribing`,
  then `answered`.
- The instruction reads something like `What is 7 * 23?`.
- The answer is `161`.
- A **waveform** is drawn under the card, and pressing play moves a playhead
  across it.
- The header shows `device connected :5000`.
- The right rail shows `utterances 1` and a non-zero `received`.

**Evidence:** all six. If the waveform is blank but the transcript is right, the
audio is being written but not served — check that `web/dist` was built and that
`GET /api/audio/<id>` returns `audio/wav`.

---

## 4. Typed input takes the same path

In the console's input box, type `what is 12 divided by 4` and press enter.

**Evidence:** a new card with that exact text, answered `3`, with no audio row
(there was no audio). This proves the LLM path is independent of the hardware.

---

## 5. The device connects

Flash the firmware and wake FRIDAY (say the wake word, or press the button
depending on the build). Watch the server log:

```
friday.ingest   device connected (#1) from ('192.168.1.42', 51234)
friday.stt      transcribed 20260926-131127-5b598ddb1abb46a2.wav -> 'What is 7 times 23?' (conf 0.62, 104ms)
```

**Evidence:** the header flips to `device connected`, and `connections` in the
right rail goes to 1 while the socket is open.

If nothing connects at all, work down: is the PC on the same subnet? Is the
ESP32's Wi-Fi actually associated? Is port 5000 the one the firmware is dialling?

---

## 6. One real utterance, end to end

Say the wake word, then **"what is the weather in Pune right now"**.

Check the card:

- Instruction: `What is the weather in Pune right now?` — **"Pune" spelled
  correctly**, not "Pooner". If it is misheard, the hotwords list in
  `config.yaml` is the thing to extend.
- Answer: a real temperature.
- Expanding the tool trace shows `get_weather(city="Pune")` and the raw result.
- The waveform matches what you said: you can hear yourself in the playback.

**Evidence:** all four. Then try **"what is 7 times 23"** and confirm `161` —
that one takes the deterministic fast path, so it should answer in well under a
second, with `llm 0ms` in the card footer.

---

## 7. Two utterances in a row

Say two questions back to back, with the wake word between them.

**Evidence:** two separate cards, `seq 1` and `seq 2`, each with its own
waveform and its own answer. No merged card, no cross-talk between the two
transcripts.

If the second is missing: the device must send **one WAV per connection** and
close the socket between them. See `docs/ingest_contract.md` §4.

---

## 8. Silence is not an answer

Say the wake word and then nothing.

**Evidence:** either no card at all (utterance below the 0.3 s floor, dropped) or
a card marked `no speech` — never a card with a fabricated answer.

---

## 9. Restart recovery

With the console open, hit the wake word, then `Ctrl-C` the server before the
answer appears. Restart it.

**Evidence:** the stranded card now reads `failed` with
`interrupted: the server restarted while this turn was in flight`. It does not
sit on `thinking` forever. Ask a new question and it answers normally.

---

## 10. Long-run behaviour

Leave it running for a few minutes of casual use, then check the right rail.

- `stored` climbs and then stops at 200 — retention is doing its job.
- Disk use stays flat: `retention.delete_audio` removes the audio of pruned turns.
- Nothing degrades; the right rail's `avg stt` and `avg llm` stay in the same
  range as the single-shot numbers.

```bash
du -sh data/
df -h . | tail -1
```

---

## 11. What to do when a step fails

| Symptom | Where to look |
|---|---|
| No card at all | `friday.ingest` lines. Is an utterance being logged? If not, the device never sent one. |
| Card stuck at `received` | The worker thread is wedged. Restart; the card becomes `failed`. |
| Card stuck at `transcribing` | STT. Check `friday.stt` for a load error, and `nvidia-smi` for VRAM. |
| Card stuck at `thinking` | The LLM. Check for an `LLM load failed` line. |
| Card at `calling tool` for 20 s+ | The weather HTTP call. The whole lookup is capped at `weather_timeout_s` × 1.5, so past that something is wrong rather than slow. |
| Right transcript, garbled waveform | The audio format is inferred from config, not from the wire, so a device that is not 16 kHz mono PCM16 transcribes as nonsense. Check `ingest.sample_rate` against `docs/ingest_contract.md` §2, and §6 for the 32-bit I2S slot trap. |
| One question produced two cards | The server's `ingest.vad_rms_threshold` is at or above the device's `STREAM_SILENCE_RMS_GATE`. The startup warning names this; see `docs/ingest_contract.md` §3. |
| Middle of the sentence missing a word | The device dropped a hop: `Stream: queue full, dropping hop` in the serial log. Not fixable PC-side — see `docs/ingest_contract.md` §3. |
| Right answer, no waveform | Frontend. `cd web && npm run build`, and check `GET /api/audio/<id>`. |
| `"Pune"` heard as `"Pooner"` | Add the word to `stt.hotwords` in `config.yaml`. |
| Answer correct but obviously not from the tools | `chat_format` must be `chatml-function-calling`; plain `chatml` makes the model hallucinate instead of calling. |
| Card says `unclear` for something you clearly said | `min_confidence` is too high for the room. Real commands measured 0.62-0.83, but a noisy room costs confidence; lower `stt.min_confidence` in `config.yaml` in 0.05 steps. |
| Card says `unclear` with a repeated phrase | Correct behaviour. Whisper loops on music and room tone, and it loops *confidently* — 0.74 on a real capture of a speaker playing music — so `is_repetitive()` in `server/stt.py` catches what no confidence threshold could. If it fires on real speech, the KWS is triggering on the TV. |

Server logs are the fastest diagnostic:

```bash
.venv/bin/python -m server --log-level debug
```
