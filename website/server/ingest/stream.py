"""Turn a TCP byte stream into discrete utterances.

The device is expected to send a complete WAV, but the server accepts several
shapes so that whatever the firmware actually emits is still usable:

* ``RIFF``/``WAVE`` stream  -- the documented contract
* headerless raw PCM16      -- endpointed with an energy VAD
* several WAVs concatenated -- split on the declared chunk boundaries
"""

from __future__ import annotations

import array
import struct
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass

from . import wav
from .wav import WavError

# A WAV header with a streaming sentinel is at least this long.
MIN_HEADER = 12
# Anything smaller than this cannot hold a usable header.
MAX_HEADER_PROBE = 4096


@dataclass(slots=True)
class Utterance:
    """One complete audio utterance, ready to be written to disk."""

    pcm: bytes
    sample_rate: int
    channels: int
    bits_per_sample: int
    # How the utterance was delimited, useful for diagnostics and tests.
    source: str

    @property
    def duration_s(self) -> float:
        block = self.channels * (self.bits_per_sample // 8)
        if not block or not self.sample_rate:
            return 0.0
        return (len(self.pcm) // block) / self.sample_rate


class StreamFramer:
    """Incremental framer. Feed bytes with :meth:`feed`, drain with :meth:`utterances`.

    Emits ``Utterance`` objects as soon as a full one is available, and (when
    the connection closes) whatever tail remains via :meth:`flush`.
    """

    def __init__(
        self,
        *,
        sample_rate: int,
        channels: int,
        bits: int,
        vad_silence_s: float = 0.8,
        vad_rms_threshold: float = 0.02,
        vad_preroll_s: float = 0.15,
        vad_hangover_s: float = 0.08,
        max_utterance_s: float = 10.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.bits = bits
        self.vad_silence_s = vad_silence_s
        self.vad_rms_threshold = vad_rms_threshold
        self.vad_preroll_s = vad_preroll_s
        self.vad_hangover_s = vad_hangover_s
        self.max_utterance_s = max_utterance_s
        self._frame_bytes = channels * (bits // 8) or 1
        self._buf = bytearray()
        # Set once the first bytes reveal what kind of stream this is.
        self._mode: str | None = None
        # VAD state, only used for headerless PCM.
        self._speaking = False
        self._silence_run = 0
        self._voice = bytearray()
        self._voice_frames = 0
        self._preroll: deque[bytes] = deque(
            maxlen=max(1, int(vad_preroll_s * sample_rate))
        )
        self.hangover_frames = int(vad_hangover_s * sample_rate)

    # ------------------------------------------------------------------ feed
    def feed(self, chunk: bytes) -> list[Utterance]:
        self._buf.extend(chunk)
        out: list[Utterance] = []

        while True:
            if self._mode is None:
                if len(self._buf) < MIN_HEADER:
                    break
                if wav.looks_like_wav(bytes(self._buf[:12])):
                    self._mode = "wav"
                    continue
                # Not a RIFF header. If the stream is going to turn out to be
                # raw PCM we cannot wait forever for certainty, so decide now.
                self._mode = "raw"
                out.extend(self._drain_raw(force=False))
                continue

            if self._mode == "wav":
                produced = self._drain_wav()
                if not produced:
                    break
                out.extend(produced)
            else:
                out.extend(self._drain_raw(force=False))
                break
        return out

    def flush(self) -> list[Utterance]:
        """End of connection: emit whatever is left, repairing lengths if needed."""
        out: list[Utterance] = []
        if self._mode == "wav":
            out.extend(self._drain_wav(final=True))
        elif self._mode == "raw":
            out.extend(self._drain_raw(force=True))
        elif self._buf:
            # Too short to identify; treat as a raw tail.
            self._mode = "raw"
            out.extend(self._drain_raw(force=True))
        self._buf.clear()
        self._preroll.clear()
        return out

    # ------------------------------------------------------------------- wav
    def _drain_wav(self, final: bool = False) -> list[Utterance]:
        """Emit every complete WAV frame currently buffered.

        Handles both back-to-back WAVs on one connection and a single WAV whose
        declared size is wrong (repaired at ``flush`` time).
        """
        out: list[Utterance] = []
        while True:
            try:
                info = wav.parse_wav_header(bytes(self._buf))
            except WavError:
                break

            declared = info.data_size
            have = len(self._buf) - info.data_offset
            # A zero or oversized declared size means "length unknown": the
            # firmware is streaming and will only ever tell us on close.
            plausible = bool(declared) and declared <= have
            if plausible and len(self._buf) >= info.data_offset + declared:
                end = info.data_offset + declared
                out.append(self._make_utterance(bytes(self._buf[:end]), info, "wav"))
                del self._buf[:end]
                continue

            if final:
                # Take whatever arrived, whatever the header claimed.
                if len(self._buf) > info.data_offset:
                    out.append(
                        self._make_utterance(bytes(self._buf), info, "wav-final")
                    )
                del self._buf[:]
            break
        return out

    def _make_utterance(
        self, blob: bytes, info: wav.WavInfo, source: str
    ) -> Utterance:
        payload = wav.normalize_to_wav(
            blob,
            default_sample_rate=self.sample_rate,
            default_channels=self.channels,
            default_bits=self.bits,
        )
        final = wav.parse_wav_header(payload)
        return Utterance(
            pcm=payload[final.data_offset :],
            sample_rate=final.sample_rate,
            channels=final.channels,
            bits_per_sample=final.bits_per_sample,
            source=source,
        )

    # ------------------------------------------------------------------- raw
    def _drain_raw(self, force: bool) -> list[Utterance]:
        """Endpoint headerless PCM with an energy VAD.

        Frames go through three states. Before any speech is seen they fill a
        bounded pre-roll buffer, so the first syllable survives. Once speech
        starts they are committed to the current utterance. A run of silence
        long enough to reach ``vad_silence_s`` ends the utterance, and the tail
        of that silence is dropped rather than spoken back to the VAD.
        """
        if self.bits != 16:
            # Only 16-bit raw PCM can be VAD'd without a real decoder.
            return self._drain_raw_blind(force)

        block = self.channels * 2
        usable = (len(self._buf) // block) * block
        silence_frames = max(1, int(self.vad_silence_s * self.sample_rate))
        max_frames = max(1, int(self.max_utterance_s * self.sample_rate))
        out: list[Utterance] = []

        if usable:
            raw = bytes(self._buf[:usable])
            del self._buf[:usable]
            for i in range(usable // block):
                frame = raw[i * block : (i + 1) * block]
                loud = _frame_rms(frame, self.channels) >= self.vad_rms_threshold

                if loud and not self._speaking:
                    # Commit the lead-in so the utterance does not start mid-word.
                    for lead in self._preroll:
                        self._voice += lead
                    self._voice_frames += len(self._preroll)
                    self._preroll.clear()
                    self._speaking = True

                if loud:
                    self._silence_run = 0
                elif self._speaking:
                    self._silence_run += 1
                    if self._silence_run >= silence_frames:
                        # The silence that ended the turn is the endpoint, not
                        # speech. Drop all of it bar a short hangover, so a
                        # word-final consonant is not clipped off.
                        hangover = min(self.hangover_frames, self._silence_run - 1)
                        out.append(
                            self._cut_utterance(
                                "raw-vad",
                                drop_tail_frames=self._silence_run - 1 - hangover,
                            )
                        )
                        continue
                else:
                    # Not speaking yet: keep only a bounded lead-in.
                    self._preroll.append(frame)
                    continue

                self._voice += frame
                self._voice_frames += 1
                if self._voice_frames >= max_frames:
                    out.append(self._cut_utterance("raw-vad-max"))

        if force and self._speaking:
            # The connection closed mid-utterance. Hand on what we have.
            out.append(self._cut_utterance("raw-vad-final"))
        elif force:
            # Only silence ever arrived; a pure-silence turn is not worth sending.
            self._preroll.clear()
        return out

    def _drain_raw_blind(self, force: bool) -> list[Utterance]:
        """No VAD available for this bit depth: cut on size alone."""
        block = self.channels * max(1, self.bits // 8)
        max_bytes = int(self.max_utterance_s * self.sample_rate) * block
        out: list[Utterance] = []
        while len(self._buf) >= max_bytes:
            out.append(
                self._utterance_from_pcm(bytes(self._buf[:max_bytes]), "raw-blind")
            )
            del self._buf[:max_bytes]
        if force and self._buf:
            out.append(self._utterance_from_pcm(bytes(self._buf), "raw-blind-final"))
            self._buf.clear()
        return out

    def _cut_utterance(self, source: str, *, drop_tail_frames: int = 0) -> Utterance:
        frames = self._voice_frames - max(0, drop_tail_frames)
        pcm = bytes(self._voice[: max(0, frames) * self._frame_bytes])
        self._voice.clear()
        self._voice_frames = 0
        self._speaking = False
        self._silence_run = 0
        return self._utterance_from_pcm(pcm, source)

    def _utterance_from_pcm(self, pcm: bytes, source: str) -> Utterance:
        return Utterance(
            pcm=pcm,
            sample_rate=self.sample_rate,
            channels=self.channels,
            bits_per_sample=self.bits,
            source=source,
        )


def _frame_rms(frame: bytes, channels: int) -> float:
    """RMS of one interleaved frame, normalised to 0..1."""
    if channels < 1:
        return 0.0
    a = array.array("h")
    a.frombytes(frame[: channels * 2])
    if not a:
        return 0.0
    total = 0
    for s in a:
        total += s * s
    return (total / len(a)) ** 0.5 / 32768.0


def frames(pcm: bytes, channels: int = 1) -> Iterator[bytes]:
    """Iterate interleaved frames of a PCM payload."""
    block = channels * 2
    for i in range(0, len(pcm) - block + 1, block):
        yield pcm[i : i + block]


def read_pcm_duration_s(pcm: bytes, sample_rate: int, channels: int, bits: int = 16) -> float:
    block = channels * max(1, bits // 8)
    if not block or not sample_rate:
        return 0.0
    return (len(pcm) // block) / sample_rate


def to_mono_pcm16(pcm: bytes, channels: int, bits: int, sample_rate: int) -> tuple[bytes, int]:
    """Best-effort downmix to mono 16-bit PCM, returning ``(pcm, sample_rate)``.

    Handles the two mistakes a raw-PCM device actually makes:
    32-bit slots where only the top 16 bits are meaningful, and stereo frames
    where the device said mono.
    """
    if bits == 16 and channels == 1:
        return pcm, sample_rate

    if bits == 32:
        usable = (len(pcm) // 4) * 4
        a = array.array("i")
        a.frombytes(pcm[:usable])
        out = array.array("h", [s >> 16 for s in a])
        if channels > 1:
            out = _downmix(out, channels)
        return out.tobytes(), sample_rate

    if bits == 24:
        usable = (len(pcm) // 3) * 3
        out = bytearray()
        for i in range(0, usable, 3):
            # 24-bit signed -> 16-bit signed
            val = int.from_bytes(pcm[i : i + 3], "little", signed=True) >> 8
            out += struct.pack("<h", max(-32768, min(32767, val)))
        return bytes(out), sample_rate

    if channels > 1 and bits == 16:
        return _downmix(array.array("h", pcm), channels).tobytes(), sample_rate

    return pcm, sample_rate


def _downmix(samples: array.array, channels: int) -> array.array:
    n = (len(samples) // channels) * channels
    out = array.array("h", bytes(2 * (n // channels)))
    for i in range(0, n, channels):
        acc = 0
        for c in range(channels):
            acc += samples[i + c]
        out[i // channels] = int(acc / channels)
    return out
