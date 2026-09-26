"""The wire contract, pinned against the firmware it mirrors.

`server/ingest/protocol.py` transcribes `#define`s out of the device firmware so
the ingest server, the replay tool and the docs all agree. That only helps if
the transcription is right and stays right, so the first test here reads the
actual firmware and fails if a constant has drifted.

The rest are the behavioural consequences of those constants. The important one
is `test_mid_sentence_pause_yields_one_turn`: the device endpointed the
utterance already, and a server whose VAD is stricter than the device's splits
one question into two turns at a natural pause.
"""

from __future__ import annotations

import re
from array import array
from pathlib import Path

import pytest

from server.config import IngestConfig
from server.ingest import protocol
from server.ingest.stream import StreamFramer

from .conftest import silence_pcm, tone_pcm

# The firmware lives next to this project during development, but the tests must
# pass on a machine that only has the PC side (and on CI).
FIRMWARE_CANDIDATES = [
    Path(__file__).resolve().parents[2] / "friday_kws" / "main" / "friday_kws.cpp",
    Path(__file__).resolve().parents[2] / "friday" / "main" / "sih_voice.cpp",
]


def _find_firmware() -> Path | None:
    for candidate in FIRMWARE_CANDIDATES:
        if candidate.is_file():
            return candidate
    return None


def _defines(path: Path) -> dict[str, str]:
    """Every ``#define NAME value`` in the file, with the value as written.

    Trailing ``//`` comments are stripped: the firmware annotates most of the
    defines that matter (``#define STREAM_SILENCE_MS 800  // stop streaming
    after this much continuous silence``).
    """
    text = path.read_text(errors="replace")
    out: dict[str, str] = {}
    pattern = re.compile(r"^\s*#define\s+(\w+)\s+(.+?)\s*$", re.MULTILINE)
    for name, value in pattern.findall(text):
        out.setdefault(name, value.split("//", 1)[0].strip())
    return out


# --------------------------------------------------------------------- drift
@pytest.mark.skipif(_find_firmware() is None, reason="firmware source not available")
def test_constants_match_the_firmware() -> None:
    """protocol.py must not drift from the device it is written against."""
    defines = _defines(_find_firmware())

    def number(name: str) -> float:
        assert name in defines, f"{name} is gone from the firmware; re-check the contract"
        # Strip a trailing f / integer suffix and any surrounding parens.
        return float(defines[name].strip().rstrip("fF").strip("()"))

    def expression(name: str) -> float:
        """Resolve the simple ``A * B`` defines the hop size is written as."""
        assert name in defines, f"{name} is gone from the firmware"
        return float(defines[name].rstrip("fF"))

    assert number("SAMPLE_RATE") == protocol.SAMPLE_RATE
    assert number("HOP_MS") == protocol.HOP_MS
    assert number("STREAM_SILENCE_MS") == protocol.SILENCE_MS
    assert number("STREAM_SILENCE_RMS_GATE") == pytest.approx(protocol.SILENCE_RMS_GATE)
    assert number("STREAM_MAX_TIME_MS") == protocol.MAX_TIME_MS
    # INPUT_GAIN is what makes the two sides' RMS comparable at all: if it is
    # not 1.0 the firmware's gate is not in the same units the server measures.
    assert number("INPUT_GAIN") == pytest.approx(1.0)

    # HOP_SAMPLES is written as (HOP_FRAMES * STRIDE) in the firmware.
    hop = defines.get("HOP_SAMPLES", "")
    if "*" in hop:
        assert expression("HOP_FRAMES") * expression("STRIDE") == protocol.HOP_SAMPLES
    else:
        assert expression("HOP_SAMPLES") == protocol.HOP_SAMPLES

    assert protocol.BYTES_PER_HOP == protocol.HOP_SAMPLES * 2
    assert protocol.PREROLL_S == 0.0, "the device sends nothing before the wake-word hop"


@pytest.mark.skipif(_find_firmware() is None, reason="firmware source not available")
def test_firmware_sends_no_container_header() -> None:
    """The device is headerless, so the framer's WAV path is the fallback.

    This is the assumption the whole contract rests on, and the one the old
    contract doc got wrong.
    """
    text = _find_firmware().read_text(errors="replace")
    assert not re.search(r'"RIFF"', text), "firmware now writes a RIFF header; update the contract"
    assert protocol.SERVER_RMS_GATE < protocol.SILENCE_RMS_GATE


# ------------------------------------------------------------------ framing
def test_firmware_hops_are_3840_bytes() -> None:
    pcm = tone_pcm(1.0)
    hops = list(protocol.firmware_hops(pcm))
    assert hops, "a loud second of audio must produce hops"
    assert all(len(h) == protocol.BYTES_PER_HOP for h in hops[:-1])
    assert len(hops) == pytest.approx(1000 / protocol.HOP_MS, abs=1)


def test_firmware_hops_stop_at_the_silence_gate() -> None:
    """Speech, then 800 ms of true silence: the device stops, and the trailing
    silence after that point is never sent."""
    spoken = tone_pcm(1.0, amplitude=9000)
    payload = b"".join(protocol.firmware_hops(spoken + silence_pcm(3.0)))
    sent_s = len(payload) / protocol.FRAME_BYTES / protocol.SAMPLE_RATE
    # ~1 s of speech plus the 800 ms hangover, quantised to 120 ms hops.
    assert 1.6 <= sent_s <= 2.0
    assert sent_s < 4.0, "the device must not stream the whole 3 s of trailing silence"


def test_firmware_hops_respect_the_ten_second_cap() -> None:
    """Ten seconds of unbroken speech is cut, even though nothing went quiet."""
    payload = b"".join(protocol.firmware_hops(tone_pcm(20.0, amplitude=9000)))
    sent_s = len(payload) / protocol.FRAME_BYTES / protocol.SAMPLE_RATE
    assert sent_s == pytest.approx(protocol.MAX_TIME_MS / 1000, abs=protocol.HOP_MS / 1000)


def test_rms_is_normalised_to_the_units_the_firmware_uses() -> None:
    # Full-scale sine is ~0.707 of full scale; the firmware compares this number
    # against 0.007, so it has to be in the same 0..1 range.
    assert protocol.rms(tone_pcm(1.0, amplitude=32767)) == pytest.approx(0.707, abs=0.01)
    assert protocol.rms(silence_pcm(0.5)) == 0.0
    # 0.010 sits above the device's gate and below a naive 0.02 server gate.
    quiet = array("h", [int(0.010 * 32768)] * protocol.SAMPLE_RATE).tobytes()
    assert protocol.SILENCE_RMS_GATE < protocol.rms(quiet) < 0.02


# --------------------------------------------------------- the split-turn bug
def _frame_with_config(cfg: IngestConfig) -> StreamFramer:
    return StreamFramer(
        sample_rate=cfg.sample_rate,
        channels=cfg.channels,
        bits=cfg.bits,
        vad_silence_s=cfg.vad_silence_s,
        vad_rms_threshold=cfg.vad_rms_threshold,
        max_utterance_s=cfg.max_utterance_s,
    )


def _paused_command() -> bytes:
    """A question with a natural beat in the middle.

    The pause is at an RMS of 0.010: above the device's 0.007 gate, so the
    device keeps streaming and sends one utterance; below a 0.02 server gate,
    so a too-strict server thinks the speaker stopped.
    """
    quiet = array("h", [int(0.010 * 32768)] * (protocol.SAMPLE_RATE)).tobytes()
    first = tone_pcm(1.0, amplitude=9000)
    second = tone_pcm(1.0, freq=330.0, amplitude=9000)
    return first + quiet + second


def test_mid_sentence_pause_yields_one_turn() -> None:
    """The regression: one question, one turn.

    With a server gate at or above the device's, this becomes two turns and the
    user gets two answers to a single question.
    """
    cfg = IngestConfig()
    assert cfg.vad_rms_threshold < protocol.SILENCE_RMS_GATE

    framer = _frame_with_config(cfg)
    turns = []
    for hop in protocol.firmware_hops(_paused_command()):
        turns += framer.feed(hop)
    turns += framer.flush()

    assert len(turns) == 1, (
        f"expected one turn, got {len(turns)}: "
        f"{[(round(t.duration_s, 2), t.source) for t in turns]}"
    )
    assert turns[0].duration_s == pytest.approx(3.0, abs=0.3)


def test_a_stricter_server_gate_really_does_split_the_turn() -> None:
    """Guard the guard: if the fix were reverted, this should fail loudly.

    Without this, `test_mid_sentence_pause_yields_one_turn` could be passing for
    an unrelated reason (a fixture that happens to have no quiet stretch in it,
    which is exactly how the original bug hid).
    """
    cfg = IngestConfig(vad_rms_threshold=0.02)
    framer = _frame_with_config(cfg)
    turns = []
    for hop in protocol.firmware_hops(_paused_command()):
        turns += framer.feed(hop)
    turns += framer.flush()

    assert len(turns) == 2, "the 0.02 gate no longer splits the turn; revisit the 0.006 default"


# ------------------------------------------------------------ drift detection
def test_check_alignment_is_clean_for_the_shipped_config() -> None:
    assert protocol.check_alignment(IngestConfig()) == []


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("vad_rms_threshold", 0.02, "STREAM_SILENCE_RMS_GATE"),
        ("vad_rms_threshold", 0.007, "STREAM_SILENCE_RMS_GATE"),
        ("vad_silence_s", 0.4, "STREAM_SILENCE_MS"),
        ("max_utterance_s", 8.0, "STREAM_MAX_TIME_MS"),
        ("sample_rate", 8000, "sample_rate"),
        ("bits", 32, "PCM16"),
        ("channels", 2, "mono"),
        ("host", "127.0.0.1", "0.0.0.0"),
    ],
)
def test_check_alignment_catches_drift(field: str, value: object, expected: str) -> None:
    cfg = IngestConfig(**{field: value})
    problems = protocol.check_alignment(cfg)
    assert problems, f"{field}={value!r} should have been flagged"
    assert any(expected in p for p in problems), problems


def test_check_alignment_catches_the_loopback_bind() -> None:
    """A 127.0.0.1 bind looks fine locally and is unreachable from the device."""
    problems = protocol.check_alignment(IngestConfig(host="127.0.0.1"))
    assert any("Wi-Fi" in p for p in problems)
