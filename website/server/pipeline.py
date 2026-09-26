"""Pipeline orchestration: audio in, answer out.

One worker thread drains a queue so the ingest socket is never blocked by
inference. Each stage publishes to the bus, which is what makes the UI show a
live pipeline rather than a spinner.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .agent import Agent
from .bus import EventBus
from .config import Config
from .db import Database
from .ingest.wav import write_wav_file
from .stt import SpeechToText
from .tools import build_registry
from .tools.calc import try_fast_path
from .tools.registry import ToolRegistry
from .tools.weather import WeatherClient

log = logging.getLogger("friday.pipeline")


@dataclass(slots=True)
class AudioJob:
    """One utterance waiting to be processed."""

    pcm: bytes
    sample_rate: int
    channels: int
    bits: int
    source: str


@dataclass(slots=True)
class TextJob:
    """A turn whose transcript is already known (typed, not spoken)."""

    turn_id: str
    text: str


class Pipeline:
    def __init__(self, cfg: Config, db: Database, bus: EventBus) -> None:
        self.cfg = cfg
        self.db = db
        self.bus = bus

        self.weather = WeatherClient(
            ttl=cfg.tools.weather_cache_ttl,
            timeout=cfg.tools.weather_timeout_s,
        )
        self.registry: ToolRegistry = build_registry(cfg, self.weather)
        self.stt = SpeechToText(cfg)
        self.agent = Agent(cfg, self.registry)

        self._queue: queue.Queue[AudioJob | TextJob | None] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._stop = threading.Event()
        self.device_connected = False

    # ------------------------------------------------------------- lifecycle
    def start(self) -> None:
        if self._worker is not None:
            return
        if self.cfg.llm.eager_load:
            # Load in the background so the HTTP port opens immediately.
            threading.Thread(target=self._warm_llm, name="llm-warm", daemon=True).start()
        self._worker = threading.Thread(target=self._run, name="pipeline", daemon=True)
        self._worker.start()
        log.info("pipeline worker started")

    def _warm_llm(self) -> None:
        try:
            self.agent.load()
        except Exception as exc:  # noqa: BLE001
            log.warning("LLM warm-up failed: %s", exc)

    def stop(self) -> None:
        self._stop.set()
        self._queue.put(None)
        if self._worker is not None:
            self._worker.join(timeout=5)
        self.agent.close()

    def _run(self) -> None:
        while not self._stop.is_set():
            job = self._queue.get()
            if job is None:
                break
            try:
                if isinstance(job, TextJob):
                    self._answer(job.turn_id, job.text)
                else:
                    self._process(job)
            except Exception:  # noqa: BLE001
                log.exception("pipeline job failed")
            finally:
                self._queue.task_done()

    # ---------------------------------------------------------------- intake
    def submit(self, job: AudioJob) -> None:
        self._queue.put(job)

    def submit_text(self, turn_id: str, text: str) -> None:
        """Queue a turn that already has its transcript (no audio involved)."""
        self._queue.put(TextJob(turn_id=turn_id, text=text))

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    # ------------------------------------------------------------- processing
    def _process(self, job: AudioJob) -> None:
        cfg = self.cfg
        turn_id = uuid.uuid4().hex[:16]

        filename = f"{time.strftime('%Y%m%d-%H%M%S')}-{turn_id}.wav"
        path = cfg.recordings_dir / filename
        write_wav_file(
            path,
            job.pcm,
            sample_rate=job.sample_rate,
            channels=job.channels,
            bits_per_sample=job.bits,
        )
        duration = _duration_s(job)

        turn = self.db.create_turn(
            id=turn_id,
            audio_path=str(path),
            duration_s=round(duration, 2),
            sample_rate=job.sample_rate,
            channels=job.channels,
            source=job.source,
        )
        self._emit("turn.created", turn)
        log.info("turn %s created from %s (%.2fs)", turn_id, job.source, duration)

        # --- transcribe -------------------------------------------------
        self._update(turn_id, state="transcribing")
        try:
            result = self.stt.transcribe(path)
        except Exception as exc:  # noqa: BLE001
            log.exception("transcription failed")
            self._update(turn_id, state="error", error=f"transcription failed: {exc}")
            return

        text = (result.text or "").strip()
        self._update(
            turn_id,
            transcript=text or None,
            confidence=result.confidence,
            transcript_ms=result.elapsed_ms,
        )

        if not text:
            self._update(turn_id, state="no_speech", error="no speech recognised")
            return
        if result.repetitive:
            # Checked before confidence on purpose. A decode loop is emitted at
            # *high* confidence -- real device audio of a speaker playing music
            # scored 0.73 with twelve copies of one place name -- so a confidence
            # threshold will never see it.
            self._update(
                turn_id, state="unclear", error="transcript repeated itself; not speech"
            )
            return
        if result.confidence < cfg.stt.min_confidence:
            self._update(turn_id, state="unclear", error="transcript too uncertain")
            return

        self._answer(turn_id, text)

    def _answer(self, turn_id: str, text: str) -> None:
        """Answer a turn whose transcript is already known."""
        cfg = self.cfg
        # A typed turn arrives with no transcript column set yet; the UI shows
        # the instruction from that column, so it has to be written either way.
        self._update(turn_id, state="thinking", transcript=text)

        # Deterministic path: exact arithmetic without the model in the loop.
        fast = try_fast_path(text) if cfg.tools.calc_fast_path else None
        if fast is not None:
            answer = f"{fast.display}."
            self._update(
                turn_id,
                state="done",
                answer=answer,
                llm_ms=0,
                tool_calls=[
                    {
                        "name": "calculate",
                        "arguments": {"expression": fast.expression},
                        "result": {"result": fast.display},
                        "error": None,
                        "duration_ms": 0,
                    }
                ],
            )
            log.info("turn %s answered by calc fast path: %s", turn_id, answer)
            self._prune()
            return

        try:
            reply = self.agent.answer(text)
        except Exception as exc:  # noqa: BLE001
            log.exception("agent failed")
            self._update(turn_id, state="error", error=f"assistant failed: {exc}")
            return

        if reply.error and not reply.text:
            self._update(turn_id, state="error", error=reply.error)
            return

        self._update(
            turn_id,
            state="done",
            answer=reply.text or None,
            llm_ms=reply.elapsed_ms,
            tool_calls=[c.to_dict() for c in reply.tool_calls],
        )
        log.info(
            "turn %s answered in %dms using %d tool call(s)",
            turn_id,
            reply.elapsed_ms,
            reply.rounds,
        )
        self._prune()

    # ---------------------------------------------------------------- events
    def _update(self, turn_id: str, event: str = "turn.updated", **fields: Any) -> None:
        turn = self.db.update_turn(turn_id, **fields)
        if turn is not None:
            self._emit(event, turn)

    def _emit(self, event: str, turn: dict) -> None:
        self.bus.publish_threadsafe(
            {"event": event, "turn": _public_turn(turn)}
        )

    def _prune(self) -> None:
        cfg = self.cfg.retention
        if cfg.max_turns <= 0 or self.db.count() <= cfg.max_turns:
            return
        for orphan in self.db.prune(cfg.max_turns, cfg.delete_audio):
            if cfg.delete_audio:
                try:
                    Path(orphan).unlink(missing_ok=True)
                except OSError as exc:
                    log.warning("could not delete pruned audio %s: %s", orphan, exc)


# ------------------------------------------------------------------ helpers


def _duration_s(job: AudioJob) -> float:
    block = max(1, job.channels * (job.bits // 8))
    if not job.sample_rate:
        return 0.0
    return (len(job.pcm) // block) / job.sample_rate


def _public_turn(turn: dict) -> dict:
    """Shape a DB row for the wire."""
    return {
        "id": turn["id"],
        "seq": turn["seq"],
        "created_at": turn["created_at"],
        "updated_at": turn["updated_at"],
        "state": turn["state"],
        "audio_url": f"/api/audio/{turn['id']}" if turn.get("audio_path") else None,
        "duration_s": turn.get("duration_s"),
        "sample_rate": turn.get("sample_rate"),
        "channels": turn.get("channels"),
        "source": turn.get("source"),
        "transcript": turn.get("transcript"),
        "confidence": turn.get("confidence"),
        "answer": turn.get("answer"),
        "error": turn.get("error"),
        "tool_calls": turn.get("tool_calls") or [],
        "transcript_ms": turn.get("transcript_ms"),
        "llm_ms": turn.get("llm_ms"),
    }
