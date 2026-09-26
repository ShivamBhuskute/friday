#!/usr/bin/env python
"""Generate the speech fixtures used by the golden STT and pipeline tests.

Uses Piper (a small local neural TTS) so the fixtures are real speech with a
known transcript, and resamples the 22.05 kHz output to the 16 kHz the device
actually streams.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import wave
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = PROJECT_ROOT / "fixtures"
TARGET_RATE = 16000

# The transcript each clip is *supposed* to contain. Whisper rarely matches
# character-for-character, so the golden test normalises before comparing.
FIXTURES_SPEC: list[tuple[str, str, str]] = [
    ("q_math_7x23.wav", "what is 7 times 23", "7 * 23"),
    ("q_math_12_div_4.wav", "what is 12 divided by 4", "12 / 4"),
    ("q_math_2_plus_2.wav", "what is two plus two", "2 + 2"),
    ("q_weather_pune.wav", "what is the weather in Pune right now", None),
    ("q_weather_delhi.wav", "what is the weather in Delhi", None),
    ("q_time_now.wav", "what time is it right now", None),
    ("q_status.wav", "what is the system status", None),
    ("q_greeting.wav", "hello there", None),
    ("jfk_excerpt.wav", "And so my fellow Americans", None),
]

PIPER_VOICE = "en_US-amy-medium"


def resample(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Linear-interpolation resample of 16-bit mono PCM."""
    if src_rate == dst_rate:
        return pcm
    n_in = len(pcm) // 2
    samples = struct.unpack(f"<{n_in}h", pcm)
    ratio = src_rate / dst_rate
    n_out = int(n_in / ratio)
    out = []
    for i in range(n_out):
        pos = i * ratio
        i0 = int(pos)
        i1 = min(i0 + 1, n_in - 1)
        frac = pos - i0
        out.append(int(samples[i0] * (1 - frac) + samples[i1] * frac))
    return struct.pack(f"<{len(out)}h", *out)


def trim_silence(pcm: bytes, threshold: int = 200) -> bytes:
    """Drop leading/trailing near-silence so Whisper is not asked to guess.

    The threshold is deliberately low: a quiet first syllable such as "hello"
    sits well below a naive cutoff and gets clipped away otherwise.
    """
    n = len(pcm) // 2
    if n == 0:
        return pcm
    samples = struct.unpack(f"<{n}h", pcm)
    start = 0
    while start < n and abs(samples[start]) < threshold:
        start += 1
    end = n - 1
    while end > start and abs(samples[end]) < threshold:
        end -= 1
    # Keep padding so the clip does not start or stop abruptly.
    pad = TARGET_RATE // 10  # 100 ms
    start = max(0, start - pad)
    end = min(n - 1, end + pad)
    return pcm[start * 2 : (end + 1) * 2]


def synthesize(text: str, out_path: Path) -> bool:
    """Render ``text`` to a 16 kHz mono WAV at ``out_path``."""
    try:
        import piper.voice  # noqa: F401  - availability probe only
    except ImportError:
        print(
            "piper-tts is not installed. Install it with:\n"
            "  uv pip install piper-tts",
            file=sys.stderr,
        )
        return False

    import io

    voice = _load_voice()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_out:
        voice.synthesize_wav(text, wav_out)

    raw = buf.getvalue()
    with wave.open(io.BytesIO(raw), "rb") as src:
        src_rate = src.getframerate()
        channels = src.getnchannels()
        width = src.getsampwidth()
        frames = src.readframes(src.getnframes())

    if channels > 1 or width != 2:
        print(f"  ! unexpected piper output: {channels}ch {width * 8}bit", file=sys.stderr)
        return False

    pcm = resample(frames, src_rate, TARGET_RATE)
    pcm = trim_silence(pcm)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_path), "wb") as dst:
        dst.setnchannels(1)
        dst.setsampwidth(2)
        dst.setframerate(TARGET_RATE)
        dst.writeframes(pcm)
    return True


_voice_cache = None


def _load_voice():
    """Download the Piper voice once and load it."""
    global _voice_cache
    if _voice_cache is not None:
        return _voice_cache

    from piper.download_voices import download_voice
    from piper.voice import PiperVoice

    voice_dir = PROJECT_ROOT / "models" / "piper"
    voice_dir.mkdir(parents=True, exist_ok=True)
    print(f"       fetching piper voice {PIPER_VOICE} (first run only)")
    download_voice(PIPER_VOICE, voice_dir)

    onnx = voice_dir / f"{PIPER_VOICE}.onnx"
    if not onnx.exists():
        raise FileNotFoundError(f"piper voice model not found at {onnx}")

    _voice_cache = PiperVoice.load(str(onnx))
    return _voice_cache


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", help="generate just the fixture whose name contains this")
    ap.add_argument("--force", action="store_true", help="regenerate existing fixtures")
    args = ap.parse_args(argv)

    FIXTURES.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict] = {}

    for name, spoken, expect in FIXTURES_SPEC:
        out = FIXTURES / name
        if out.exists() and not args.force:
            print(f"[skip] {name}")
        else:
            print(f"[tts ] {name}: {spoken!r}")
            if not synthesize(spoken, out):
                return 1
        with wave.open(str(out), "rb") as w:
            rate, frames = w.getframerate(), w.getnframes()
        manifest[name] = {
            "spoken": spoken,
            # Words that must survive normalisation, in order.
            "expect_contains": _expectation(spoken, expect),
            "duration_s": round(frames / rate, 2),
            "sample_rate": rate,
        }
        print(f"       {manifest[name]['duration_s']}s @ {rate}Hz")

    (FIXTURES / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nwrote {FIXTURES / 'manifest.json'}")
    return 0


def _expectation(spoken: str, normalised: str | None) -> list[str]:
    """The key words the transcript must contain after normalisation."""
    if normalised:
        return [t for t in normalised.split() if t not in {"*", "/", "+"}]
    keep = [
        "weather", "time", "system", "status", "pune", "delhi", "hello",
        "fellow", "americans", "right", "now",
    ]
    return [w for w in spoken.lower().split() if w in keep]


if __name__ == "__main__":
    sys.exit(main())
