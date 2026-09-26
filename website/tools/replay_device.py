#!/usr/bin/env python
"""Replay the real FRIDAY firmware protocol over TCP.

This is not a hypothetical device: it reproduces, byte for byte, what
``friday_kws/main/friday_kws.cpp`` puts on the wire, so the pipeline can be
exercised end to end on a laptop. The framing comes from
``server/ingest/protocol.py``, which mirrors the firmware's ``#define``s, so
this tool and the ingest server cannot disagree about the contract.

What the device actually does, and therefore what this does by default:

* connects to ``ingest.port`` when the wake word fires
* sends **headerless** 16 kHz mono PCM16 in 3840-byte hops, no RIFF header
* stops sending after 800 ms of audio below an RMS of 0.007, or after 10 s
* closes the socket, which is how the server knows the utterance ended

Examples::

    # the default: exactly what the device would put on the wire
    python tools/replay_device.py fixtures/q_math_7x23.wav

    # a question with a natural mid-sentence pause -- this is the case that
    # a too-strict server VAD splits into two turns
    python tools/replay_device.py --gap-ms 900 fixtures/q_weather_pune.wav

    # send a real RIFF/WAV instead, to check that code path still works
    python tools/replay_device.py --wav fixtures/q_weather_pune.wav

    # a synthetic tone, for checking the listener without real speech
    python tools/replay_device.py --sine 2.0
"""

from __future__ import annotations

import argparse
import math
import socket
import struct
import sys
import time
from array import array
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from server.ingest import protocol  # noqa: E402
from server.ingest.wav import WavError, build_wav_header, parse_wav_header  # noqa: E402


def read_wav(path: Path) -> tuple[bytes, bytes]:
    """Return ``(header_including_data_chunk_size, pcm)`` from a WAV file."""
    raw = path.read_bytes()
    info = parse_wav_header(raw)
    pcm = raw[info.data_offset : info.data_offset + info.data_size]
    header = raw[: info.data_offset + 8]
    return header, pcm


def synthesize_tone(seconds: float, sample_rate: int = protocol.SAMPLE_RATE) -> bytes:
    """A quiet hum, enough to pass the VAD and exercise the pipeline."""
    n = int(seconds * sample_rate)
    samples = (int(9000 * math.sin(2 * math.pi * 220 * i / sample_rate)) for i in range(n))
    return struct.pack(f"<{n}h", *samples)


def insert_gap(pcm: bytes, gap_ms: int, level: float = 0.010) -> bytes:
    """Splice a quiet stretch into the middle of ``pcm``.

    ``level`` is a normalised 0..1 amplitude and defaults to 0.010, which is
    the interesting case: above the device's 0.007 gate, so the device keeps
    streaming, but below a naively-chosen 0.020 server gate, so a too-strict
    server thinks the speaker stopped. That gap is what a real person does
    between "...the weather" and "in Pune".
    """
    n = int(protocol.SAMPLE_RATE * gap_ms / 1000)
    filler = array("h", (int(level * 32768) for _ in range(n))).tobytes()
    cut = len(pcm) // 2 & ~(protocol.FRAME_BYTES - 1)
    return pcm[:cut] + filler + pcm[cut:]


def send(
    host: str,
    port: int,
    payload: bytes,
    *,
    chunk: int,
    chunk_delay: float,
    hold_open: float,
    verbose: bool,
) -> None:
    """Stream ``payload`` to the ingest port and close, as the device does."""
    if verbose:
        print(f"-> connecting to {host}:{port}")
    sock = socket.create_connection((host, port), timeout=10)
    try:
        total = len(payload)
        sent = 0
        while sent < total:
            block = payload[sent : sent + chunk]
            sock.sendall(block)
            sent += len(block)
            if verbose:
                print(f"   {sent}/{total} bytes", end="\r")
            if chunk_delay:
                time.sleep(chunk_delay)
        if verbose:
            print(f"   {sent}/{total} bytes sent")
        if hold_open:
            time.sleep(hold_open)
    finally:
        sock.shutdown(socket.SHUT_WR)
        sock.close()
        if verbose:
            print("-> connection closed")


def firmware_payload(pcm: bytes) -> tuple[bytes, float]:
    """Frame ``pcm`` the way the device would, and report what it kept.

    The device only sends audio up to the point where it gave up, so applying
    that decision here is what makes this a faithful replay instead of "the
    whole file, eventually". Returns ``(payload, seconds_sent)``.
    """
    hops = list(protocol.firmware_hops(pcm))
    return b"".join(hops), len(b"".join(hops)) / protocol.FRAME_BYTES / protocol.SAMPLE_RATE


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("wav", nargs="?", help="WAV file to replay")
    # Same thing, spelled the way the docs do. Reading "--file x" is a natural
    # thing to type when the alternative is a bare path, and a confusing
    # "unrecognized arguments" is a bad first impression of the whole project.
    ap.add_argument("--file", dest="wav_flag", metavar="WAV", help="alias for the positional WAV path")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--sine", type=float, metavar="SECONDS", help="send a synthetic tone instead")
    ap.add_argument(
        "--gap-ms",
        type=int,
        default=0,
        metavar="MS",
        help="splice a quiet gap into the middle, simulating a mid-sentence pause",
    )
    ap.add_argument(
        "--wav",
        action="store_true",
        help="send a real RIFF/WAV instead of the headerless PCM the device sends",
    )
    ap.add_argument("--chunk", type=int, default=protocol.BYTES_PER_HOP, help="bytes per send")
    ap.add_argument("--chunk-delay", type=float, default=0.0, help="seconds between sends")
    ap.add_argument("--hold", type=float, default=0.0, help="keep the socket open this long after")
    ap.add_argument("--count", type=int, default=1, help="send the payload N times")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    if args.wav and args.wav_flag:
        ap.error("give the WAV once, either positionally or with --file")
    source = args.wav_flag or args.wav

    if args.sine is not None:
        pcm = synthesize_tone(args.sine)
    elif source:
        path = Path(source)
        if not path.exists():
            print(f"no such file: {path}", file=sys.stderr)
            return 1
        try:
            _header, pcm = read_wav(path)
        except WavError as exc:
            print(f"{path} is not a usable WAV: {exc}", file=sys.stderr)
            return 1
    else:
        ap.error("provide a WAV path or --sine SECONDS")

    if args.gap_ms:
        pcm = insert_gap(pcm, args.gap_ms)

    if args.wav:
        payload = (
            build_wav_header(
                sample_rate=protocol.SAMPLE_RATE,
                channels=protocol.CHANNELS,
                bits_per_sample=protocol.BITS,
                data_size=len(pcm),
            )
            + pcm
        )
        if not args.quiet:
            print(f"   mode: RIFF/WAV ({len(payload)} bytes)")
    else:
        payload, sent_s = firmware_payload(pcm)
        if not args.quiet:
            offered = len(pcm) / protocol.FRAME_BYTES / protocol.SAMPLE_RATE
            print(
                f"   mode: headerless PCM16, {protocol.BYTES_PER_HOP}-byte hops "
                f"({protocol.HOP_MS} ms each) -- sending {sent_s:.2f}s of {offered:.2f}s"
            )
            if sent_s < offered - 0.12:
                print("   (the device stops at its silence gate / 10s cap)")

    for i in range(args.count):
        if args.count > 1 and not args.quiet:
            print(f"-- send {i + 1}/{args.count}")
        send(
            args.host,
            args.port,
            payload,
            chunk=args.chunk,
            chunk_delay=args.chunk_delay,
            hold_open=args.hold,
            verbose=not args.quiet,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
