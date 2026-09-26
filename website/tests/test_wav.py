"""WAV header construction, parsing and repair."""

from __future__ import annotations

import struct

import pytest

from server.ingest import wav
from server.ingest.wav import WavError

from .conftest import tone_pcm, wav_bytes


class TestBuildHeader:
    def test_canonical_44_byte_header(self) -> None:
        header = wav.build_wav_header(
            sample_rate=16000, channels=1, bits_per_sample=16, data_size=1000
        )
        assert len(header) == 44
        assert header[:4] == b"RIFF"
        assert header[8:12] == b"WAVE"
        assert struct.unpack_from("<I", header, 4)[0] == 36 + 1000
        assert header[12:16] == b"fmt "
        fmt = struct.unpack_from("<HHIIHH", header, 20)
        assert fmt == (1, 1, 16000, 32000, 2, 16)
        assert header[36:40] == b"data"
        assert struct.unpack_from("<I", header, 40)[0] == 1000

    def test_stereo_byte_rate(self) -> None:
        header = wav.build_wav_header(
            sample_rate=48000, channels=2, bits_per_sample=16, data_size=400
        )
        byte_rate, block_align = struct.unpack_from("<I", header, 28)[0], struct.unpack_from(
            "<H", header, 32
        )[0]
        assert byte_rate == 48000 * 4
        assert block_align == 4

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"bits_per_sample": 12},
            {"channels": 0},
            {"sample_rate": 0, "data_size": 0},
        ],
    )
    def test_rejects_impossible_formats(self, kwargs: dict) -> None:
        base = {
            "sample_rate": 16000,
            "channels": 1,
            "bits_per_sample": 16,
            "data_size": 10,
        }
        base.update(kwargs)
        with pytest.raises(WavError):
            wav.build_wav_header(**base)

    def test_negative_data_size(self) -> None:
        with pytest.raises(WavError):
            wav.build_wav_header(
                sample_rate=16000, channels=1, bits_per_sample=16, data_size=-1
            )


class TestSniff:
    def test_detects_wav(self) -> None:
        assert wav.looks_like_wav(wav_bytes(tone_pcm(0.1)))

    def test_rejects_raw_pcm(self) -> None:
        assert not wav.looks_like_wav(tone_pcm(0.1))

    def test_rejects_short_input(self) -> None:
        assert not wav.looks_like_wav(b"RI")


class TestParseHeader:
    def test_round_trip(self) -> None:
        pcm = tone_pcm(0.5)
        info = wav.parse_wav_header(wav_bytes(pcm))
        assert (info.sample_rate, info.channels, info.bits_per_sample) == (16000, 1, 16)
        assert info.data_size == len(pcm)
        assert info.duration_s == pytest.approx(0.5)
        assert info.frame_count == len(pcm) // 2

    def test_skips_unknown_chunks(self) -> None:
        """A LIST chunk before fmt (as some recorders emit) must not break us."""
        base = wav_bytes(tone_pcm(0.1))
        fmt = base[12:36]  # "fmt " + size + body
        data = base[36:]
        junk = b"LIST" + struct.pack("<I", 10) + b"INFOabcdef"
        blob = base[:12] + junk + fmt + data
        info = wav.parse_wav_header(blob)
        assert info.sample_rate == 16000
        assert blob[info.data_offset : info.data_offset + info.data_size] == tone_pcm(0.1)

    def test_extensible_format_resolves_pcm(self) -> None:
        """WAVE_FORMAT_EXTENSIBLE hides PCM in a GUID; unwrap it to plain PCM."""
        pcm = tone_pcm(0.1)
        sub_format_pcm = b"\x01\x00\x00\x00\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71"
        fmt_body = struct.pack(
            "<HHIIHHHHI", 0xFFFE, 1, 16000, 32000, 2, 16, 22, 16, 4
        ) + sub_format_pcm
        assert len(fmt_body) == 40
        blob = (
            b"RIFF"
            + struct.pack("<I", 4 + 8 + len(fmt_body) + 8 + len(pcm))
            + b"WAVE"
            + b"fmt "
            + struct.pack("<I", len(fmt_body))
            + fmt_body
            + b"data"
            + struct.pack("<I", len(pcm))
            + pcm
        )
        info = wav.parse_wav_header(blob)
        assert info.audio_format == wav.WAVE_FORMAT_PCM
        assert (info.sample_rate, info.channels, info.bits_per_sample) == (16000, 1, 16)
        assert blob[info.data_offset :] == pcm

    @pytest.mark.parametrize(
        "blob",
        [
            b"short",
            b"RIFX" + b"\x00" * 20,
            b"RIFF" + struct.pack("<I", 4) + b"AVI ",
            b"RIFF" + struct.pack("<I", 4) + b"WAVE",
        ],
    )
    def test_rejects_malformed(self, blob: bytes) -> None:
        with pytest.raises(WavError):
            wav.parse_wav_header(blob)

    def test_rejects_zero_rate(self) -> None:
        header = bytearray(wav_bytes(tone_pcm(0.1)))
        struct.pack_into("<I", header, 24, 0)  # sample rate
        with pytest.raises(WavError, match="zero sample rate"):
            wav.parse_wav_header(bytes(header))


class TestSizeRepair:
    def test_patches_both_size_fields(self) -> None:
        header = wav.build_wav_header(
            sample_rate=16000, channels=1, bits_per_sample=16, data_size=0
        )
        fixed = wav.patch_riff_sizes(header, 4096)
        assert struct.unpack_from("<I", fixed, 4)[0] == 36 + 4096
        assert struct.unpack_from("<I", fixed, 40)[0] == 4096

    def test_rejects_short_header(self) -> None:
        with pytest.raises(WavError):
            wav.patch_riff_sizes(b"RIFF" + b"\x00" * 20, 100)

    def test_normalize_repairs_zero_length(self) -> None:
        """Firmware that streams without knowing the length sends data size 0."""
        pcm = tone_pcm(0.25)
        header = wav.build_wav_header(
            sample_rate=16000, channels=1, bits_per_sample=16, data_size=0
        )
        out = wav.normalize_to_wav(
            header + pcm, default_sample_rate=16000, default_channels=1, default_bits=16
        )
        info = wav.parse_wav_header(out)
        assert info.data_size == len(pcm)
        assert out[info.data_offset :] == pcm

    def test_normalize_repairs_oversized_declaration(self) -> None:
        """A 0xFFFFFFFF streaming sentinel must be replaced with the truth."""
        pcm = tone_pcm(0.25)
        header = bytearray(wav.build_wav_header(sample_rate=16000, channels=1, bits_per_sample=16, data_size=0))
        struct.pack_into("<I", header, 4, 0xFFFFFFFF)  # riff size sentinel
        struct.pack_into("<I", header, 40, 0xFFFFFFFF)  # data size sentinel
        out = wav.normalize_to_wav(
            bytes(header) + pcm, default_sample_rate=16000, default_channels=1, default_bits=16
        )
        info = wav.parse_wav_header(out)
        assert info.data_size == len(pcm)
        assert info.duration_s == pytest.approx(0.25)
        assert out[info.data_offset :] == pcm

    def test_normalize_truncates_trailing_partial_frame(self) -> None:
        pcm = tone_pcm(0.2)
        truncated = pcm[:-1]  # a dangling byte
        header = wav.build_wav_header(
            sample_rate=16000, channels=1, bits_per_sample=16, data_size=0
        )
        out = wav.normalize_to_wav(
            header + truncated, default_sample_rate=16000, default_channels=1, default_bits=16
        )
        info = wav.parse_wav_header(out)
        # 16-bit mono: a whole frame is 2 bytes, so the odd byte is dropped.
        assert info.data_size == (len(pcm) - 1) // 2 * 2
        assert info.data_size % 2 == 0
        assert out[info.data_offset :] == truncated[: info.data_size]

    def test_normalize_keeps_a_plausible_declaration(self) -> None:
        pcm = tone_pcm(0.2)
        out = wav.normalize_to_wav(
            wav_bytes(pcm), default_sample_rate=16000, default_channels=1, default_bits=16
        )
        info = wav.parse_wav_header(out)
        assert info.data_size == len(pcm)


class TestWriteFile:
    def test_writes_a_playable_file(self, tmp_path) -> None:
        pcm = tone_pcm(0.3)
        path = wav.write_wav_file(
            tmp_path / "sub" / "a.wav", pcm, sample_rate=16000, channels=1, bits_per_sample=16
        )
        assert path.exists()
        info = wav.parse_wav_header(path.read_bytes())
        assert info.duration_s == pytest.approx(0.3)

    def test_playable_by_soundfile(self, tmp_path) -> None:
        """Guard against a header that is structurally valid but not decodable."""
        soundfile = pytest.importorskip("soundfile")
        pcm = tone_pcm(0.3)
        path = wav.write_wav_file(
            tmp_path / "b.wav", pcm, sample_rate=16000, channels=1, bits_per_sample=16
        )
        data, rate = soundfile.read(str(path), dtype="int16")
        assert rate == 16000
        assert len(data) == len(pcm) // 2


class TestPcmStats:
    def test_silence(self) -> None:
        stats = wav.pcm_stats(b"\x00\x00" * 1600, sample_rate=16000)
        assert stats["rms"] == 0.0
        assert stats["duration_s"] == pytest.approx(0.1)
        assert not stats["clipped"]

    def test_detects_clipping(self) -> None:
        pcm = struct.pack("<1600h", *([32767] * 1600))
        stats = wav.pcm_stats(pcm, sample_rate=16000)
        assert stats["clipped"]
        assert stats["peak"] == pytest.approx(1.0, abs=1e-3)

    def test_tone_rms_is_sane(self) -> None:
        stats = wav.pcm_stats(tone_pcm(1.0, amplitude=9000), sample_rate=16000)
        assert 0.15 < stats["rms"] < 0.35
        assert not stats["clipped"]

    def test_empty_payload(self) -> None:
        assert wav.pcm_stats(b"", sample_rate=16000)["rms"] == 0.0
