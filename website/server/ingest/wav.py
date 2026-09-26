"""RIFF/WAVE construction, parsing and repair.

The device is expected to send a standard 16 kHz mono PCM16 WAV, but real
firmware often cannot know the payload length before streaming it. Everything
here is built so that a header with a placeholder ``data`` size still yields a
valid file once the payload has actually been received.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

WAVE_FORMAT_PCM = 0x0001
WAVE_FORMAT_EXTENSIBLE = 0xFFFE
RIFF_MAGIC = b"RIFF"
WAVE_MAGIC = b"WAVE"
DATA_MAGIC = b"data"


class WavError(ValueError):
    """Raised when a byte stream is not a usable WAVE payload."""


@dataclass(slots=True)
class WavInfo:
    """Parsed shape of a WAVE stream."""

    sample_rate: int
    channels: int
    bits_per_sample: int
    audio_format: int
    data_offset: int
    data_size: int

    @property
    def block_align(self) -> int:
        return self.channels * (self.bits_per_sample // 8)

    @property
    def frame_count(self) -> int:
        if not self.block_align:
            return 0
        return self.data_size // self.block_align

    @property
    def duration_s(self) -> float:
        if not self.sample_rate:
            return 0.0
        return self.frame_count / self.sample_rate


def build_wav_header(
    *,
    sample_rate: int,
    channels: int,
    bits_per_sample: int,
    data_size: int,
    audio_format: int = WAVE_FORMAT_PCM,
) -> bytes:
    """Build a canonical 44-byte RIFF/WAVE header."""
    if bits_per_sample % 8:
        raise WavError(f"bits_per_sample must be a multiple of 8, got {bits_per_sample}")
    if channels < 1:
        raise WavError(f"channels must be >= 1, got {channels}")
    if sample_rate < 1:
        raise WavError(f"sample_rate must be >= 1, got {sample_rate}")
    if data_size < 0:
        raise WavError(f"data_size must be >= 0, got {data_size}")

    block_align = channels * (bits_per_sample // 8)
    byte_rate = sample_rate * block_align
    # A streaming writer may legitimately declare a sentinel length; the field
    # is 32-bit, so clamp rather than overflow the struct pack.
    riff_size = (36 + data_size) & 0xFFFFFFFF
    return b"".join(
        [
            RIFF_MAGIC,
            struct.pack("<I", riff_size),
            WAVE_MAGIC,
            b"fmt ",
            struct.pack("<I", 16),
            struct.pack("<H", audio_format),
            struct.pack("<H", channels),
            struct.pack("<I", sample_rate),
            struct.pack("<I", byte_rate),
            struct.pack("<H", block_align),
            struct.pack("<H", bits_per_sample),
            DATA_MAGIC,
            struct.pack("<I", data_size),
        ]
    )


def looks_like_wav(data: bytes) -> bool:
    """Cheap sniff: does this stream start with a RIFF/WAVE header?"""
    return len(data) >= 12 and data[:4] == RIFF_MAGIC and data[8:12] == WAVE_MAGIC


def parse_wav_header(data: bytes) -> WavInfo:
    """Parse the leading header of a WAVE stream.

    Tolerates unknown/odd chunks before ``data`` (LIST, fact, etc.) by walking
    the chunk list. Only the ``fmt `` and ``data`` chunks are required.
    """
    if len(data) < 12:
        raise WavError(f"stream too short for a WAVE header: {len(data)} bytes")
    if data[:4] != RIFF_MAGIC:
        raise WavError("missing RIFF magic")
    if data[8:12] != WAVE_MAGIC:
        raise WavError("missing WAVE magic (not a RIFF/WAVE file)")

    pos = 12
    fmt: tuple[int, int, int, int] | None = None  # fmt, channels, rate, bits
    data_pos: int | None = None
    data_len = 0

    while pos + 8 <= len(data):
        chunk_id = data[pos : pos + 4]
        (chunk_size,) = struct.unpack("<I", data[pos + 4 : pos + 8])
        body = pos + 8

        if chunk_id == b"fmt ":
            if chunk_size < 16 or body + 16 > len(data):
                raise WavError("truncated or undersized fmt chunk")
            audio_format, channels, rate, _byte_rate, _align, bits = struct.unpack(
                "<HHIIHH", data[body : body + 16]
            )
            if audio_format == WAVE_FORMAT_EXTENSIBLE and chunk_size >= 40:
                # The real format lives in the GUID's first two bytes.
                (audio_format,) = struct.unpack("<H", data[body + 24 : body + 26])
            fmt = (audio_format, channels, rate, bits)
        elif chunk_id == DATA_MAGIC:
            data_pos = body
            # A declared size of 0xFFFFFFFF (streaming) or 0 (unknown) is common
            # from firmware; both mean "use what actually arrives".
            data_len = chunk_size
            break

        pos = body + chunk_size + (chunk_size & 1)  # chunks are word-aligned

    if fmt is None:
        raise WavError("no fmt chunk found")
    if data_pos is None:
        raise WavError("no data chunk found")

    audio_format, channels, rate, bits = fmt
    if rate == 0:
        raise WavError("fmt chunk declares a zero sample rate")
    if channels == 0:
        raise WavError("fmt chunk declares zero channels")

    return WavInfo(
        sample_rate=rate,
        channels=channels,
        bits_per_sample=bits,
        audio_format=audio_format,
        data_offset=data_pos,
        data_size=data_len,
    )


def patch_riff_sizes(header: bytes, actual_data_size: int) -> bytes:
    """Rewrite the RIFF and data sizes in an existing header.

    Used when the device declared a placeholder (or streaming) length and then
    simply stopped sending. ``header`` must be at least ``data_offset + 8``
    bytes so both size fields are addressable.
    """
    if len(header) < 44:
        raise WavError("header too short to patch; expected at least 44 bytes")
    out = bytearray(header)
    struct.pack_into("<I", out, 4, 36 + actual_data_size)
    struct.pack_into("<I", out, len(out) - 4, actual_data_size)
    return bytes(out)


def normalize_to_wav(
    payload: bytes,
    *,
    default_sample_rate: int,
    default_channels: int,
    default_bits: int,
) -> bytes:
    """Turn a received WAVE stream into a self-consistent WAV file on disk.

    ``payload`` is the raw concatenation of header + PCM as received. The
    declared ``data`` size is authoritative when it is plausible (does not
    exceed what arrived); otherwise it is repaired from the byte count.
    """
    info = parse_wav_header(payload)
    pcm = payload[info.data_offset :]

    declared = info.data_size
    # A declared length is only trustworthy if the payload actually contains it.
    if declared == 0 or declared > len(pcm):
        pcm = pcm[: len(pcm) - (len(pcm) % info.block_align)]
        header = payload[: info.data_offset]
        header = patch_riff_sizes(header, len(pcm))
        return header + pcm

    pcm = pcm[:declared]
    return payload[: info.data_offset] + pcm


def write_wav_file(
    path: Path,
    pcm: bytes,
    *,
    sample_rate: int,
    channels: int,
    bits_per_sample: int,
) -> Path:
    """Write PCM payload to ``path`` with a freshly built header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = build_wav_header(
        sample_rate=sample_rate,
        channels=channels,
        bits_per_sample=bits_per_sample,
        data_size=len(pcm),
    )
    path.write_bytes(header + pcm)
    return path


def pcm_stats(pcm: bytes, *, sample_rate: int, channels: int = 1, bits: int = 16) -> dict:
    """Cheap quality metrics used to reject silence and detect clipping."""
    if bits != 16:
        return {"rms": 0.0, "peak": 0.0, "duration_s": 0.0, "clipped": False}
    count = len(pcm) // 2
    if count == 0:
        return {"rms": 0.0, "peak": 0.0, "duration_s": 0.0, "clipped": False}
    samples = struct.unpack(f"<{count}h", pcm[: count * 2])
    total = 0
    peak = 0
    for s in samples:
        total += s * s
        a = s if s >= 0 else -s
        if a > peak:
            peak = a
    return {
        "rms": (total / count) ** 0.5 / 32768.0,
        "peak": peak / 32768.0,
        "duration_s": count / sample_rate,
        # 0dBFS-ish: a full-scale square wave is a classic 32-bit-slot mistake
        "clipped": peak > 32000,
    }
