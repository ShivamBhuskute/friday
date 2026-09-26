"""The wire contract, transcribed from the firmware.

Every constant here mirrors a ``#define`` in the device firmware
(``friday_kws/main/friday_kws.cpp``). This module exists so the ingest
server, the replay tool and the docs cannot drift apart: they all read these
numbers instead of hard-coding their own.

The short version, which is *not* what the old contract doc claimed:

* The device sends **headerless raw PCM16**, not a WAV file. There is no
  RIFF header, and no pre-roll -- the first sample sent is the first sample
  of the hop *after* the one the wake word was detected in.
* **One TCP connection per utterance.** It connects when the wake word
  fires, streams hops until it has seen enough silence, then closes the
  socket. The close is the end-of-utterance signal.
* The device does its own endpointing, so the server's VAD must be
  *looser* than the device's gate. If the server is stricter it will cut an
  utterance in half at a natural mid-sentence pause and answer twice.
"""

from __future__ import annotations

from array import array
from collections.abc import Iterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import IngestConfig

# --------------------------------------------------------------------- device
# #define SAMPLE_RATE 16000
SAMPLE_RATE = 16000
# #define HOP_FRAMES 6 / #define STRIDE 320  ->  #define HOP_SAMPLES (HOP_FRAMES * STRIDE)
HOP_SAMPLES = 1920
# #define HOP_MS 120
HOP_MS = 120
# #define STREAM_SILENCE_MS 800
SILENCE_MS = 800
# #define STREAM_SILENCE_RMS_GATE 0.007f, with #define INPUT_GAIN 1.0f the RMS
# is already normalised to 0..1, i.e. the same units the server measures in.
SILENCE_RMS_GATE = 0.007
# #define STREAM_MAX_TIME_MS 10000
MAX_TIME_MS = 10_000

CHANNELS = 1
BITS = 16
FRAME_BYTES = CHANNELS * (BITS // 8)
BYTES_PER_HOP = HOP_SAMPLES * FRAME_BYTES

#: The device sends nothing at all before the wake-word hop, so there is no
#: pre-roll to preserve. Do not expect leading audio.
PREROLL_S = 0.0

#: The server's own silence gate. Deliberately *below* the device's: the device
#: already endpointed the utterance, so the server's VAD is only a safety net
#: for a device that dies mid-sentence. Sitting above the device's gate makes
#: the server cut first, which splits one question into two turns.
SERVER_RMS_GATE = 0.006


def rms(pcm: bytes) -> float:
    """Normalised 0..1 RMS of a PCM16 payload, matching the firmware's math.

    Same formula as ``hop_rms`` in the firmware, so a number computed here
    means the same thing as a number computed there.
    """
    usable = len(pcm) - (len(pcm) % FRAME_BYTES)
    if usable <= 0:
        return 0.0
    samples = array("h")
    samples.frombytes(pcm[:usable])
    if not samples:
        return 0.0
    total = sum(s * s for s in samples)
    return (total / len(samples)) ** 0.5 / 32768.0


def firmware_hops(pcm: bytes) -> Iterator[bytes]:
    """Yield the exact byte hops the firmware would put on the wire.

    Mirrors the device's send loop: fixed-size hops, and the stream stops as
    soon as there has been ``SILENCE_MS`` of audio below ``SILENCE_RMS_GATE``
    or ``MAX_TIME_MS`` has elapsed. Anything the device would not have sent
    (the speech after it gave up) is not yielded, so this is a faithful
    replay rather than "the whole file, eventually".
    """
    silence_ms = 0
    total_ms = 0
    for offset in range(0, len(pcm), BYTES_PER_HOP):
        hop = pcm[offset : offset + BYTES_PER_HOP]
        yield hop
        silence_ms = silence_ms + HOP_MS if rms(hop) < SILENCE_RMS_GATE else 0
        total_ms += HOP_MS
        if silence_ms >= SILENCE_MS or total_ms >= MAX_TIME_MS:
            return


def spoken_duration_s(pcm: bytes) -> float:
    """How much audio ``firmware_hops`` would actually transmit, in seconds."""
    sent = sum(len(h) for h in firmware_hops(pcm))
    return sent / FRAME_BYTES / SAMPLE_RATE


def check_alignment(cfg: IngestConfig) -> list[str]:
    """Return human-readable mismatches between ``cfg`` and the firmware.

    Empty means the server and the device agree. This exists because the two
    sides are edited by different people on different machines, and a silent
    disagreement shows up as "it transcribed half my sentence", which is a
    miserable thing to debug from a log.
    """
    problems: list[str] = []

    if cfg.sample_rate != SAMPLE_RATE:
        problems.append(
            f"sample_rate {cfg.sample_rate} but the device sends {SAMPLE_RATE}; "
            "transcription will be garbage"
        )
    if cfg.channels != CHANNELS:
        problems.append(f"channels {cfg.channels} but the device sends {CHANNELS} (mono)")
    if cfg.bits != BITS:
        problems.append(f"bits {cfg.bits} but the device sends {BITS} (PCM16)")

    if cfg.vad_rms_threshold >= SILENCE_RMS_GATE:
        problems.append(
            f"vad_rms_threshold {cfg.vad_rms_threshold} is at or above the "
            f"device's STREAM_SILENCE_RMS_GATE ({SILENCE_RMS_GATE}); the server "
            "will cut utterances at natural mid-sentence pauses and produce "
            f"more than one turn per question. Use {SERVER_RMS_GATE}."
        )

    if cfg.vad_silence_s * 1000 < SILENCE_MS:
        problems.append(
            f"vad_silence_s {cfg.vad_silence_s} is shorter than the device's "
            f"STREAM_SILENCE_MS ({SILENCE_MS / 1000}); the server will "
            "endpoint before the device has finished speaking"
        )

    if cfg.max_utterance_s * 1000 < MAX_TIME_MS:
        problems.append(
            f"max_utterance_s {cfg.max_utterance_s} is shorter than the "
            f"device's STREAM_MAX_TIME_MS ({MAX_TIME_MS / 1000}); the server "
            "will split a long utterance"
        )

    if cfg.host not in ("0.0.0.0", "::", ""):
        problems.append(
            f"ingest host is {cfg.host}; the device connects over Wi-Fi, so "
            "this must be 0.0.0.0 or the device cannot reach the server"
        )

    return problems
