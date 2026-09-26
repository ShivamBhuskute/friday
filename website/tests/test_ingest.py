"""The TCP ingest path, over a real socket.

Everything below the pipeline is exercised for real: a listening socket, byte
stream framing, the duration gates, and connection teardown. The pipeline itself
is stubbed so no model is needed.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from server.config import Config
from server.ingest.server import IngestServer
from server.ingest.wav import build_wav_header, parse_wav_header
from server.pipeline import AudioJob

from .conftest import silence_pcm, tone_pcm


class RecordingPipeline:
    """Collects the jobs the ingest server hands over."""

    def __init__(self) -> None:
        self.jobs: list[AudioJob] = []
        self.device_connected = False

    def submit(self, job: AudioJob) -> None:
        self.jobs.append(job)

    @property
    def queue_depth(self) -> int:
        return len(self.jobs)


@pytest.fixture
async def ingest(cfg: Config):
    cfg.ingest.host = "127.0.0.1"
    cfg.ingest.port = 0  # let the OS pick
    pipeline = RecordingPipeline()
    server = IngestServer(cfg, pipeline)
    await server.start()
    server.pipeline = pipeline
    try:
        yield server, pipeline, cfg
    finally:
        await server.stop()


async def send(payload: bytes, server: IngestServer, *, size: int = 4096, close: bool = True):
    """Write ``payload`` to the ingest port in ``size``-byte slices."""
    reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
    for i in range(0, len(payload), size):
        writer.write(payload[i : i + size])
        await writer.drain()
    if close:
        writer.close()
        try:
            await writer.wait_closed()
        except OSError:
            pass
    else:
        return reader, writer


async def wait_for_jobs(pipeline: RecordingPipeline, count: int, timeout: float = 5.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if len(pipeline.jobs) >= count:
            return pipeline.jobs
        await asyncio.sleep(0.01)
    raise AssertionError(f"expected {count} job(s), got {len(pipeline.jobs)}")


class TestWavIngest:
    async def test_a_wav_arrives_as_one_job(self, ingest) -> None:
        server, pipeline, _ = ingest
        pcm = tone_pcm(1.0, amplitude=12000)
        blob = build_wav_header(
            sample_rate=16000, channels=1, bits_per_sample=16, data_size=len(pcm)
        ) + pcm
        await send(blob, server)
        jobs = await wait_for_jobs(pipeline, 1)
        assert jobs[0].pcm == pcm
        assert jobs[0].sample_rate == 16000
        assert jobs[0].channels == 1
        assert jobs[0].source == "wav"

    @pytest.mark.parametrize("slice_size", [1, 7, 512, 8192, 1_000_000])
    async def test_tcp_packet_boundaries_do_not_matter(self, ingest, slice_size) -> None:
        """The device's send() size must not change what the server hears."""
        server, pipeline, _ = ingest
        pcm = tone_pcm(0.5, amplitude=12000)
        blob = build_wav_header(
            sample_rate=16000, channels=1, bits_per_sample=16, data_size=len(pcm)
        ) + pcm
        await send(blob, server, size=slice_size)
        jobs = await wait_for_jobs(pipeline, 1)
        assert jobs[0].pcm == pcm

    async def test_two_utterances_on_one_connection(self, ingest) -> None:
        server, pipeline, _ = ingest
        a, b = tone_pcm(0.5, freq=200.0, amplitude=12000), tone_pcm(0.5, freq=600.0, amplitude=12000)
        blob = b"".join(
            build_wav_header(sample_rate=16000, channels=1, bits_per_sample=16, data_size=len(p))
            + p
            for p in (a, b)
        )
        await send(blob, server)
        jobs = await wait_for_jobs(pipeline, 2)
        assert [j.pcm for j in jobs] == [a, b]

    async def test_zero_length_header_is_repaired(self, ingest) -> None:
        """Firmware that cannot know the length up front sends 0 and closes."""
        server, pipeline, _ = ingest
        pcm = tone_pcm(0.6, amplitude=12000)
        blob = build_wav_header(
            sample_rate=16000, channels=1, bits_per_sample=16, data_size=0
        ) + pcm
        await send(blob, server)
        jobs = await wait_for_jobs(pipeline, 1)
        assert jobs[0].pcm == pcm
        assert jobs[0].source == "wav-final"

    async def test_separate_connections_are_independent(self, ingest) -> None:
        server, pipeline, _ = ingest
        for freq in (200.0, 400.0, 800.0):
            pcm = tone_pcm(0.5, freq=freq, amplitude=12000)
            blob = build_wav_header(
                sample_rate=16000, channels=1, bits_per_sample=16, data_size=len(pcm)
            ) + pcm
            await send(blob, server)
        jobs = await wait_for_jobs(pipeline, 3)
        assert len({j.pcm for j in jobs}) == 3


class TestRawPcmIngest:
    async def test_raw_pcm_is_endpointed_by_energy(self, ingest) -> None:
        server, pipeline, cfg = ingest
        cfg.ingest.vad_silence_s = 0.4
        payload = tone_pcm(0.5, amplitude=12000) + silence_pcm(0.6)
        await send(payload, server)
        jobs = await wait_for_jobs(pipeline, 1)
        assert jobs[0].source.startswith("raw-vad")
        # 0.5s of tone plus 0.6s of silence was sent; the endpoint silence must
        # not be carried into the turn, or the STT model hears dead air.
        assert len(jobs[0].pcm) // 2 / 16000 < 0.65

    async def test_pure_silence_is_dropped(self, ingest) -> None:
        server, pipeline, _ = ingest
        await send(silence_pcm(2.0), server)
        await asyncio.sleep(0.2)
        assert pipeline.jobs == []

    async def test_connection_close_salvages_a_partial_utterance(self, ingest) -> None:
        server, pipeline, _ = ingest
        await send(tone_pcm(0.6, amplitude=12000), server)
        jobs = await wait_for_jobs(pipeline, 1)
        assert jobs[0].source == "raw-vad-final"
        assert len(jobs[0].pcm) == int(0.6 * 16000) * 2


class TestDurationGates:
    async def test_too_short_is_dropped(self, ingest) -> None:
        server, pipeline, cfg = ingest
        cfg.ingest.min_utterance_s = 0.3
        pcm = tone_pcm(0.05, amplitude=12000)  # a click, not a word
        blob = build_wav_header(
            sample_rate=16000, channels=1, bits_per_sample=16, data_size=len(pcm)
        ) + pcm
        await send(blob, server)
        await asyncio.sleep(0.2)
        assert pipeline.jobs == []

    async def test_exactly_at_the_minimum_is_kept(self, ingest) -> None:
        server, pipeline, cfg = ingest
        cfg.ingest.min_utterance_s = 0.3
        pcm = tone_pcm(0.3, amplitude=12000)
        blob = build_wav_header(
            sample_rate=16000, channels=1, bits_per_sample=16, data_size=len(pcm)
        ) + pcm
        await send(blob, server)
        await wait_for_jobs(pipeline, 1)

    async def test_absurdly_long_is_dropped(self, ingest) -> None:
        server, pipeline, cfg = ingest
        cfg.ingest.max_utterance_s = 1.0
        pcm = tone_pcm(4.0, amplitude=12000)
        blob = build_wav_header(
            sample_rate=16000, channels=1, bits_per_sample=16, data_size=len(pcm)
        ) + pcm
        await send(blob, server)
        await asyncio.sleep(0.3)
        assert pipeline.jobs == []


class TestConnectionHandling:
    async def test_counters_and_stats(self, ingest) -> None:
        server, pipeline, _ = ingest
        pcm = tone_pcm(0.5, amplitude=12000)
        blob = build_wav_header(
            sample_rate=16000, channels=1, bits_per_sample=16, data_size=len(pcm)
        ) + pcm
        await send(blob, server)
        await wait_for_jobs(pipeline, 1)
        stats = server.stats()
        assert stats["listening"] is True
        assert stats["bytes_received"] == len(blob)
        assert stats["utterances_received"] == 1
        assert stats["last_audio_at"] is not None

    async def test_empty_connection_is_harmless(self, ingest) -> None:
        server, pipeline, _ = ingest
        await send(b"", server)
        await asyncio.sleep(0.15)
        assert pipeline.jobs == []
        assert server.active_connections == 0

    async def test_garbage_does_not_produce_a_turn(self, ingest) -> None:
        server, pipeline, _ = ingest
        await send(b"not audio at all, just text" * 40, server)
        await asyncio.sleep(0.2)
        # Headerless garbage is VAD'd and is silent, so nothing should surface.
        assert pipeline.jobs == []

    async def test_abrupt_reset_is_survived(self, ingest) -> None:
        """A yanked power cable mid-utterance must not wedge the listener."""
        server, pipeline, _ = ingest
        pcm = tone_pcm(0.5, amplitude=12000)
        blob = build_wav_header(
            sample_rate=16000, channels=1, bits_per_sample=16, data_size=0
        ) + pcm
        reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
        writer.write(blob[: len(blob) // 2])
        await writer.drain()
        writer.transport.abort()  # RST rather than FIN
        await asyncio.sleep(0.3)
        # The listener is still accepting.
        assert server.active_connections == 0
        await send(blob, server)
        await wait_for_jobs(pipeline, 1)

    async def test_concurrent_devices(self, ingest) -> None:
        server, pipeline, _ = ingest

        async def one(freq: float) -> None:
            pcm = tone_pcm(0.5, freq=freq, amplitude=12000)
            blob = build_wav_header(
                sample_rate=16000, channels=1, bits_per_sample=16, data_size=len(pcm)
            ) + pcm
            await send(blob, server)

        await asyncio.gather(*(one(200.0 * (i + 1)) for i in range(4)))
        jobs = await wait_for_jobs(pipeline, 4)
        assert len({j.pcm for j in jobs}) == 4

    async def test_stopping_closes_the_listener(self, cfg: Config) -> None:
        pipeline = RecordingPipeline()
        server = IngestServer(cfg, pipeline)
        await server.start()
        port = server.port
        await server.stop()
        with pytest.raises(OSError):
            _, writer = await asyncio.open_connection("127.0.0.1", port)
            await asyncio.sleep(0.1)


class TestRecorder:
    """The bytes handed on must be writable as a playable file."""

    async def test_job_pcm_becomes_a_valid_wav(self, tmp_path: Path, ingest) -> None:
        from server.ingest.wav import write_wav_file

        server, pipeline, _ = ingest
        pcm = tone_pcm(0.4, amplitude=12000)
        blob = build_wav_header(
            sample_rate=16000, channels=1, bits_per_sample=16, data_size=len(pcm)
        ) + pcm
        await send(blob, server)
        jobs = await wait_for_jobs(pipeline, 1)
        job = jobs[0]
        path = write_wav_file(
            tmp_path / "out.wav",
            job.pcm,
            sample_rate=job.sample_rate,
            channels=job.channels,
            bits_per_sample=job.bits,
        )
        info = parse_wav_header(path.read_bytes())
        assert info.sample_rate == 16000
        assert info.duration_s == pytest.approx(0.4, abs=0.01)
