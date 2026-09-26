"""Framing a TCP byte stream into utterances.

The interesting cases are the ones real firmware produces: arbitrary TCP
segmentation, a length field that is wrong, several WAVs back to back, and
headerless PCM that has to be endpointed by energy.
"""

from __future__ import annotations

import pytest

from server.ingest.stream import (
    StreamFramer,
    Utterance,
    read_pcm_duration_s,
    to_mono_pcm16,
)
from server.ingest.wav import build_wav_header

from .conftest import silence_pcm, tone_pcm, wav_bytes


def make_framer(**kwargs) -> StreamFramer:
    params = {"sample_rate": 16000, "channels": 1, "bits": 16}
    params.update(kwargs)
    return StreamFramer(**params)


def drain(framer: StreamFramer, data: bytes, size: int) -> list[Utterance]:
    """Feed ``data`` in ``size``-byte slices, then flush, as the socket reader would."""
    out: list[Utterance] = []
    for i in range(0, len(data), size):
        out.extend(framer.feed(data[i : i + size]))
    out.extend(framer.flush())
    return out


class TestWavStream:
    def test_single_wav(self) -> None:
        pcm = tone_pcm(0.5)
        got = drain(make_framer(), wav_bytes(pcm), size=4096)
        assert len(got) == 1
        assert got[0].pcm == pcm
        assert got[0].sample_rate == 16000
        assert got[0].source == "wav"
        assert got[0].duration_s == pytest.approx(0.5)

    @pytest.mark.parametrize("size", [1, 2, 3, 7, 43, 44, 45, 4096, 1_000_000])
    def test_tcp_segmentation_is_irrelevant(self, size: int) -> None:
        """TCP is a byte stream: the framing must not depend on packet boundaries."""
        pcm = tone_pcm(0.2)
        got = drain(make_framer(), wav_bytes(pcm), size=size)
        assert len(got) == 1
        assert got[0].pcm == pcm

    def test_two_wavs_on_one_connection(self) -> None:
        """A device that keeps the socket open for a second utterance."""
        a, b = tone_pcm(0.15, freq=200.0), tone_pcm(0.15, freq=400.0)
        blob = wav_bytes(a) + wav_bytes(b)
        got = drain(make_framer(), blob, size=512)
        assert len(got) == 2
        assert got[0].pcm == a
        assert got[1].pcm == b

    def test_partial_wav_is_withheld_until_complete(self) -> None:
        pcm = tone_pcm(0.2)
        blob = wav_bytes(pcm)
        framer = make_framer()
        assert framer.feed(blob[:-100]) == []  # incomplete, must not emit
        got = framer.feed(blob[-100:])
        assert len(got) == 1
        assert got[0].pcm == pcm

    def test_declared_length_repaired_at_close(self) -> None:
        """Firmware that does not know the length sends 0, then closes."""
        pcm = tone_pcm(0.3)
        blob = build_wav_header(
            sample_rate=16000, channels=1, bits_per_sample=16, data_size=0
        ) + pcm
        got = drain(make_framer(), blob, size=1024)
        assert len(got) == 1
        assert got[0].pcm == pcm

    def test_non_16k_rate_is_preserved(self) -> None:
        pcm = tone_pcm(0.1, sample_rate=48000)
        got = drain(make_framer(sample_rate=16000), wav_bytes(pcm, sample_rate=48000), size=999)
        assert got[0].sample_rate == 48000

    def test_empty_stream_yields_nothing(self) -> None:
        assert drain(make_framer(), b"", size=64) == []

    def test_header_only_yields_nothing(self) -> None:
        header = build_wav_header(
            sample_rate=16000, channels=1, bits_per_sample=16, data_size=0
        )
        assert drain(make_framer(), header, size=64) == []


class TestRawPcm:
    def test_vad_splits_speech_then_silence(self) -> None:
        speech = tone_pcm(0.4, amplitude=12000)
        pause = silence_pcm(1.0)
        framer = make_framer(vad_silence_s=0.5)
        got = drain(framer, speech + pause, size=1600)
        assert len(got) == 1
        assert got[0].source == "raw-vad"
        # The tail of the silence is the endpoint, not part of the utterance.
        assert got[0].duration_s < 0.9
        assert got[0].duration_s > 0.3

    def test_vad_keeps_leading_silence_out(self) -> None:
        lead = silence_pcm(0.3)
        speech = tone_pcm(0.3, amplitude=12000)
        pause = silence_pcm(0.9)
        framer = make_framer(vad_silence_s=0.5)
        got = drain(framer, lead + speech + pause, size=800)
        assert len(got) == 1
        assert got[0].duration_s < 0.6  # not the 1.5s we sent

    def test_two_utterances_split_by_silence(self) -> None:
        first = tone_pcm(0.2, freq=300.0, amplitude=12000)
        gap = silence_pcm(0.9)
        second = tone_pcm(0.2, freq=500.0, amplitude=12000)
        framer = make_framer(vad_silence_s=0.5)
        got = drain(framer, first + gap + second, size=1600)
        assert len(got) == 2
        # The gap ends the first turn; the socket closing ends the second.
        assert got[0].source == "raw-vad"
        assert got[1].source == "raw-vad-final"
        assert got[0].duration_s == pytest.approx(0.2 + 0.08, abs=0.02)

    def test_flush_emits_unterminated_speech(self) -> None:
        """A device that just stops talking must still produce an utterance."""
        framer = make_framer(vad_silence_s=10.0)
        got = drain(framer, tone_pcm(0.3, amplitude=12000), size=1600)
        assert len(got) == 1
        assert got[0].source == "raw-vad-final"

    def test_pure_silence_produces_nothing(self) -> None:
        framer = make_framer(vad_silence_s=0.5)
        assert drain(framer, silence_pcm(2.0), size=3200) == []

    def test_max_utterance_forces_a_cut(self) -> None:
        """A stuck-open microphone must not produce an unbounded utterance."""
        framer = make_framer(vad_silence_s=30.0, max_utterance_s=0.5)
        got = framer.feed(tone_pcm(1.2, amplitude=12000))
        assert len(got) >= 2
        assert all(u.duration_s <= 0.6 for u in got)

    def test_odd_trailing_byte_is_dropped(self) -> None:
        framer = make_framer(vad_silence_s=30.0)
        got = drain(framer, tone_pcm(0.2, amplitude=12000) + b"\x01", size=1024)
        assert len(got) == 1
        assert len(got[0].pcm) % 2 == 0

    def test_32bit_raw_falls_back_to_size_cutting(self) -> None:
        """Without a 16-bit view we cannot VAD, so cut on length alone."""
        pcm = b"\x00\x00\x00\x00" * 16000  # 1s of 32-bit silence
        framer = make_framer(bits=32, max_utterance_s=0.5)
        got = drain(framer, pcm, size=4096)
        assert got
        assert all(u.source.startswith("raw-blind") for u in got)


class TestDownmix:
    def test_mono_16bit_is_a_passthrough(self) -> None:
        pcm = tone_pcm(0.05)
        assert to_mono_pcm16(pcm, 1, 16, 16000) == (pcm, 16000)

    def test_stereo_16bit_is_averaged(self) -> None:
        left, right = tone_pcm(0.05, freq=200.0), tone_pcm(0.05, freq=200.0)
        interleaved = bytearray()
        for i in range(0, len(left), 2):
            interleaved += left[i : i + 2] + right[i : i + 2]
        out, rate = to_mono_pcm16(bytes(interleaved), 2, 16, 16000)
        assert rate == 16000
        # Mono means half the bytes, and a perfect L==R duplicate of the tone
        # must come back out unchanged.
        assert len(out) == len(left)
        assert out == left

    def test_32bit_takes_the_top_16_bits(self) -> None:
        """The classic I2S 32-bit-slot mistake: only the high half is signal."""
        import struct as _struct

        pcm = tone_pcm(0.05)
        wide = b"".join(
            _struct.pack("<i", s << 16) for (s,) in _struct.iter_unpack("<h", pcm)
        )
        out, _ = to_mono_pcm16(wide, 1, 32, 16000)
        assert len(out) == len(pcm)
        # Recovering the tone proves the low half was not garbage we hoped away.
        for (a,), (b,) in zip(
            _struct.iter_unpack("<h", out), _struct.iter_unpack("<h", pcm), strict=True
        ):
            assert abs(a - b) <= 2

    def test_24bit_is_scaled_down(self) -> None:
        import struct as _struct

        pcm = tone_pcm(0.05)
        wide = b"".join(_struct.pack("<i", s << 8)[0:3] for (s,) in _struct.iter_unpack("<h", pcm))
        out, _ = to_mono_pcm16(wide, 1, 24, 16000)
        assert len(out) == len(pcm)
        for (a,), (b,) in zip(
            _struct.iter_unpack("<h", out), _struct.iter_unpack("<h", pcm), strict=True
        ):
            assert abs(a - b) <= 2


class TestHelpers:
    @pytest.mark.parametrize(
        "n_bytes,sample_rate,channels,bits,expected",
        [
            (32000, 16000, 1, 16, 1.0),  # 1.0s mono 16-bit
            (32000, 16000, 2, 16, 0.5),  # 2s of stereo frames
            (32000, 16000, 1, 32, 0.5),  # 0.5s of 32-bit frames
            (0, 16000, 1, 16, 0.0),
            (1999, 16000, 1, 16, 0.0624375),  # a partial frame is not counted
        ],
    )
    def test_duration(self, n_bytes, sample_rate, channels, bits, expected) -> None:
        pcm = b"\x00" * n_bytes
        assert read_pcm_duration_s(pcm, sample_rate, channels, bits) == pytest.approx(expected)

    def test_duration_handles_zero_rate(self) -> None:
        assert read_pcm_duration_s(b"\x00\x00", 0, 1, 16) == 0.0

    def test_utterance_duration_uses_block_align(self) -> None:
        u = Utterance(
            pcm=b"\x00" * 32000, sample_rate=16000, channels=1, bits_per_sample=16, source="wav"
        )
        assert u.duration_s == pytest.approx(1.0)
