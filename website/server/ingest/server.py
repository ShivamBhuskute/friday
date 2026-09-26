"""TCP ingest listener.

Accepts connections from the device, frames them into utterances and hands each
one to the pipeline.

The device sends headerless 16 kHz mono PCM16 in 3840-byte hops, one TCP
connection per utterance, and closes the socket to signal the end. See
``server/ingest/protocol.py`` for the full contract and the constants it
mirrors from the firmware; WAV input is also accepted, so a WAV-sending
device (or the replay tool's ``--wav`` mode) still works.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from ..config import Config
from . import protocol
from .stream import StreamFramer, Utterance

if TYPE_CHECKING:
    from ..pipeline import Pipeline

log = logging.getLogger("friday.ingest")

# Devices are chatty but not fast; a long idle timeout is fine and keeps
# sockets from churning between utterances.
READ_CHUNK = 8192
IDLE_TIMEOUT_S = 300.0
PEER_LIMIT = 8


class IngestServer:
    def __init__(self, cfg: Config, pipeline: Pipeline) -> None:
        self.cfg = cfg
        self.pipeline = pipeline
        self._server: asyncio.AbstractServer | None = None
        self._peers = 0
        self._conn_seq = 0
        self.bytes_received = 0
        self.utterances_received = 0
        self.last_audio_at: float | None = None

    # ------------------------------------------------------------- lifecycle
    async def start(self) -> None:
        host, port = self.cfg.ingest.host, self.cfg.ingest.port
        self._warn_on_misalignment()
        self._server = await asyncio.start_server(self._handle, host, port)
        # Log the port that was actually bound, not the one requested: with
        # port 0 the OS picks, and "listening on :0" is useless to read.
        log.info("ingest listening on %s:%d", host, self.port)

    def _warn_on_misalignment(self) -> None:
        """Compare the ingest config against the firmware's own constants.

        The two sides are edited on different machines, and when they disagree
        the symptom is a half-transcribed sentence rather than an error, so it
        is worth saying loudly at startup instead.
        """
        for problem in protocol.check_alignment(self.cfg.ingest):
            log.warning("ingest config does not match the firmware: %s", problem)

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    @property
    def port(self) -> int:
        """Actual bound port, useful when the configured port is 0 in tests."""
        if self._server is None or not self._server.sockets:
            return self.cfg.ingest.port
        return int(self._server.sockets[0].getsockname()[1])

    @property
    def active_connections(self) -> int:
        return self._peers

    def stats(self) -> dict:
        return {
            "listening": self._server is not None,
            "port": self.port,
            "connections": self._peers,
            "bytes_received": self.bytes_received,
            "utterances_received": self.utterances_received,
            "last_audio_at": self.last_audio_at,
        }

    # ------------------------------------------------------------ connection
    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        if self._peers >= self.cfg.ingest.max_connections or self._peers >= PEER_LIMIT:
            log.warning("refusing connection: peer limit reached")
            writer.close()
            try:
                await writer.wait_closed()
            except (OSError, asyncio.CancelledError):
                pass
            return

        self._conn_seq += 1
        conn_id = self._conn_seq
        self._peers += 1
        peer = writer.get_extra_info("peername")
        log.info("device connected (#%d) from %s", conn_id, peer)

        framer = StreamFramer(
            sample_rate=self.cfg.ingest.sample_rate,
            channels=self.cfg.ingest.channels,
            bits=self.cfg.ingest.bits,
            vad_silence_s=self.cfg.ingest.vad_silence_s,
            vad_rms_threshold=self.cfg.ingest.vad_rms_threshold,
            max_utterance_s=self.cfg.ingest.max_utterance_s,
        )
        received = 0
        try:
            while True:
                try:
                    chunk = await asyncio.wait_for(
                        reader.read(READ_CHUNK), timeout=IDLE_TIMEOUT_S
                    )
                except TimeoutError:
                    log.info("connection #%d idle timeout, closing", conn_id)
                    break
                if not chunk:
                    break
                received += len(chunk)
                self.bytes_received += len(chunk)
                for utt in framer.feed(chunk):
                    self._submit(utt, conn_id)
        except (ConnectionResetError, BrokenPipeError):
            log.info("device #%d reset the connection", conn_id)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            log.exception("connection #%d failed", conn_id)
        finally:
            for utt in framer.flush():
                self._submit(utt, conn_id)
            self._peers -= 1
            log.info(
                "device #%d disconnected after %d bytes (%d utterance(s))",
                conn_id,
                received,
                self.utterances_received,
            )
            writer.close()
            try:
                await writer.wait_closed()
            except (OSError, asyncio.CancelledError):
                pass

    def _submit(self, utt: Utterance, conn_id: int) -> None:
        duration = utt.duration_s
        if duration < self.cfg.ingest.min_utterance_s:
            log.info("dropping %.2fs utterance from #%d (too short)", duration, conn_id)
            return
        if duration > self.cfg.ingest.max_utterance_s * 1.5:
            log.info("dropping %.2fs utterance from #%d (too long)", duration, conn_id)
            return

        self.utterances_received += 1
        self.last_audio_at = time.time()
        log.info(
            "utterance %d from #%d: %.2fs %dHz/%dch/%dbit via %s",
            self.utterances_received,
            conn_id,
            duration,
            utt.sample_rate,
            utt.channels,
            utt.bits_per_sample,
            utt.source,
        )
        from ..pipeline import AudioJob

        self.pipeline.submit(
            AudioJob(
                pcm=utt.pcm,
                sample_rate=utt.sample_rate,
                channels=utt.channels,
                bits=utt.bits_per_sample,
                source=utt.source,
            )
        )
