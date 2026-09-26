"""The pipeline: audio in, transcript out, answer out.

The models are stubbed so the state machine, the event stream, the retention
policy and the fast path are all exercised deterministically. The real models
are covered by ``test_stt.py`` and ``test_agent.py``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from server.agent import AgentReply
from server.bus import EventBus
from server.config import Config
from server.db import Database
from server.ingest.wav import parse_wav_header
from server.pipeline import AudioJob, Pipeline

from .conftest import tone_pcm


class FakeSTT:
    """Stands in for faster-whisper: returns a queued transcript."""

    def __init__(self, cfg: Config, script: dict[str, str] | None = None) -> None:
        self.cfg = cfg
        self.device = "cpu"
        self.compute_type = "int8"
        self._loaded = False
        self.script = script or {}
        self.seen: list[Path] = []
        self.raise_on_next: Exception | None = None
        self.result: Any = None

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        self._loaded = True

    def transcribe(self, audio_path: Path):
        from server.stt import Transcript, normalize_transcript

        self.seen.append(audio_path)
        if self.raise_on_next is not None:
            exc, self.raise_on_next = self.raise_on_next, None
            raise exc
        if self.result is not None:
            return self.result
        # Key the script on the audio duration so a test can steer the answer.
        probe = parse_wav_header(audio_path.read_bytes())
        seconds = round(probe.duration_s, 1)
        text = self.script.get(f"{seconds}", self.script.get("*", "what is 7 times 23"))
        return Transcript(
            text=normalize_transcript(text),
            confidence=0.95,
            duration_s=probe.duration_s,
            elapsed_ms=100,
        )


class FakeAgent:
    """Stands in for the LLM, recording what it was asked."""

    def __init__(self) -> None:
        self._loaded = False
        self.questions: list[str] = []
        self.reply = AgentReply(text="161.", tool_calls=[])
        self.raise_on_next: Exception | None = None

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        self._loaded = True

    def close(self) -> None:
        pass

    def answer(self, question: str) -> AgentReply:
        if self.raise_on_next is not None:
            exc, self.raise_on_next = self.raise_on_next, None
            raise exc
        self.questions.append(question)
        return self.reply


@pytest.fixture
def rig(cfg: Config, bus: EventBus):
    """A started pipeline with stubbed models, plus a collector for its events."""
    cfg.llm.eager_load = False
    db = Database(cfg.db_path)
    pipe = Pipeline(cfg, db, bus)
    pipe.stt = FakeSTT(cfg)
    pipe.agent = FakeAgent()

    class Collector:
        def __init__(self) -> None:
            self.events: list[dict] = []

        def states(self, turn_id: str) -> list[str]:
            return [
                e["turn"]["state"]
                for e in self.events
                if e.get("turn", {}).get("id") == turn_id
            ]

        def of(self, turn_id: str) -> list[dict]:
            return [e for e in self.events if e.get("turn", {}).get("id") == turn_id]

    collector = Collector()
    original_publish = bus.publish_threadsafe

    def spy(event: dict) -> None:
        collector.events.append(event)
        original_publish(event)

    bus.publish_threadsafe = spy  # type: ignore[method-assign]
    pipe.start()
    try:
        yield pipe, db, collector
    finally:
        pipe.stop()
        db.close()
        bus.publish_threadsafe = original_publish  # type: ignore[method-assign]


def wait_for(db: Database, turn_id: str, states: set[str], timeout: float = 10.0) -> dict:
    """Block until the worker thread moves a turn into one of ``states``."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        turn = db.get_turn(turn_id)
        if turn and turn["state"] in states:
            return turn
        time.sleep(0.01)
    raise AssertionError(
        f"turn {turn_id} never reached {states}; last state was "
        f"{db.get_turn(turn_id)['state'] if db.get_turn(turn_id) else 'missing'}"
    )


TERMINAL = {"done", "error", "no_speech", "unclear"}


def submit_audio(pipe: Pipeline, seconds: float = 0.5, **job_fields) -> None:
    """Queue one utterance. Audio-shape kwargs go to the job, not the tone."""
    pcm_kwargs = {
        k: job_fields[k] for k in ("sample_rate", "freq", "amplitude") if k in job_fields
    }
    pipe.submit(
        AudioJob(
            pcm=tone_pcm(seconds, **pcm_kwargs),
            sample_rate=job_fields.get("sample_rate", 16000),
            channels=job_fields.get("channels", 1),
            bits=job_fields.get("bits", 16),
            source=job_fields.get("source", "wav"),
        )
    )


def last_turn(db: Database) -> dict:
    turn = db.latest_turn()
    assert turn is not None
    return turn


def submit_and_wait(pipe: Pipeline, db: Database, seconds: float = 0.5, **job_fields) -> str:
    """Queue audio and return the id of the turn the worker creates for it.

    The turn row is written by the worker thread, so the id is not knowable at
    submit time; this polls for the row that was not there a moment ago.
    """
    before = {t["id"] for t in db.list_turns(limit=500)}
    submit_audio(pipe, seconds, **job_fields)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        fresh = [t for t in db.list_turns(limit=500) if t["id"] not in before]
        if fresh:
            return max(fresh, key=lambda t: t["seq"])["id"]
        time.sleep(0.005)
    raise AssertionError("the worker never created a turn")


class TestHappyPath:
    def test_audio_becomes_a_answered_turn(self, rig) -> None:
        pipe, db, collector = rig
        pipe.stt.script = {"*": "what is 7 times 23"}
        pipe.agent.reply = AgentReply(text="161.", tool_calls=[])

        turn = wait_for(db, submit_and_wait(pipe, db), TERMINAL)

        assert turn["state"] == "done"
        assert turn["transcript"] == "what is 7 * 23"
        assert turn["answer"] == "161."
        assert turn["duration_s"] == pytest.approx(0.5, abs=0.01)
        assert turn["sample_rate"] == 16000

    def test_audio_file_is_written_and_playable(self, rig) -> None:
        pipe, db, _ = rig
        turn = wait_for(db, submit_and_wait(pipe, db, seconds=0.4), TERMINAL)
        path = Path(turn["audio_path"])
        assert path.exists()
        assert path.suffix == ".wav"
        info = parse_wav_header(path.read_bytes())
        assert info.sample_rate == 16000
        assert info.channels == 1
        assert info.duration_s == pytest.approx(0.4, abs=0.01)

    def test_state_sequence_is_observable(self, rig) -> None:
        """The UI renders progress from these events, so their order matters."""
        pipe, db, collector = rig
        turn = wait_for(db, submit_and_wait(pipe, db), TERMINAL)
        states = collector.states(turn["id"])
        assert states[0] == "uploaded"
        assert states[-1] == "done"
        for earlier, later in zip(states, states[1:], strict=False):
            order = ["uploaded", "transcribing", "thinking", "done"]
            assert order.index(earlier) <= order.index(later), states

    def test_created_event_is_published(self, rig) -> None:
        pipe, db, collector = rig
        turn = wait_for(db, submit_and_wait(pipe, db), TERMINAL)
        created = [e for e in collector.of(turn["id"]) if e["event"] == "turn.created"]
        assert created
        assert created[0]["turn"]["audio_url"] == f"/api/audio/{turn['id']}"

    def test_published_turn_never_leaks_the_filesystem_path(self, rig) -> None:
        """A turn is broadcast to browsers; an absolute path is not wanted."""
        pipe, db, collector = rig
        wait_for(db, submit_and_wait(pipe, db), TERMINAL)
        for event in collector.events:
            assert "audio_path" not in event["turn"]
            assert event["turn"]["audio_url"].startswith("/api/audio/")


class TestFastPath:
    def test_arithmetic_skips_the_model(self, rig) -> None:
        pipe, db, _ = rig
        pipe.stt.script = {"*": "what is 7 times 23"}
        turn = wait_for(db, submit_and_wait(pipe, db), TERMINAL)

        assert turn["answer"] == "161."
        assert turn["llm_ms"] == 0
        assert pipe.agent.questions == []  # the LLM was never asked
        assert turn["tool_calls"][0]["name"] == "calculate"
        assert turn["tool_calls"][0]["result"]["result"] == "161"

    def test_non_arithmetic_reaches_the_model(self, rig) -> None:
        pipe, db, _ = rig
        pipe.stt.script = {"*": "what is the weather in Pune"}
        pipe.agent.reply = AgentReply(text="It is 28 degrees.", tool_calls=[])
        turn = wait_for(db, submit_and_wait(pipe, db), TERMINAL)
        assert pipe.agent.questions == ["what is the weather in Pune"]
        assert turn["llm_ms"] is not None

    def test_fast_path_can_be_disabled(self, rig) -> None:
        pipe, db, _ = rig
        pipe.cfg.tools.calc_fast_path = False
        pipe.stt.script = {"*": "what is 7 times 23"}
        wait_for(db, submit_and_wait(pipe, db), TERMINAL)
        assert pipe.agent.questions == ["what is 7 * 23"]


class TestNoSpeech:
    def test_empty_transcript_becomes_no_speech(self, rig) -> None:
        pipe, db, _ = rig
        pipe.stt.script = {"*": "uh friday um erm"}
        turn = wait_for(db, submit_and_wait(pipe, db), TERMINAL)
        assert turn["state"] == "no_speech"
        assert pipe.agent.questions == []
        assert turn["answer"] is None

    def test_low_confidence_short_transcript_is_unclear(self, rig) -> None:
        pipe, db, _ = rig
        pipe.stt.result = _transcript("hm", confidence=0.05, elapsed_ms=80)
        turn = wait_for(db, submit_and_wait(pipe, db), TERMINAL)
        assert turn["state"] == "unclear"
        assert "uncertain" in turn["error"]

    def test_a_long_uncertain_transcript_is_also_rejected(self, rig) -> None:
        """The confidence gate used to apply only to text under four characters.

        That inverted the intent: it trusted long transcripts and distrusted
        short ones, so a rambling low-confidence hallucination was answered as
        though it were a question. A real capture of room tone produced
        "So do need a cool. I don't know this hour" at confidence 0.28.
        """
        pipe, db, _ = rig
        pipe.stt.result = _transcript(
            "So do need a cool. I don't know this hour", confidence=0.28, elapsed_ms=80
        )
        turn = wait_for(db, submit_and_wait(pipe, db), TERMINAL)
        assert turn["state"] == "unclear"
        assert pipe.agent.questions == []

    def test_a_confident_repetition_loop_is_still_rejected(self, rig) -> None:
        """Confidence cannot catch this one, so repetition has to.

        A real capture of a speaker playing music scored 0.73 with twelve
        consecutive copies of one place name. At 0.73 it passes any confidence
        gate, and the LLM would have been asked to answer the loop.
        """
        pipe, db, _ = rig
        pipe.stt.result = _transcript(
            "1,2,3,4 get on the dance floor " + "Kolkata " * 12,
            confidence=0.73,
            elapsed_ms=80,
            repetitive=True,
        )
        turn = wait_for(db, submit_and_wait(pipe, db), TERMINAL)
        assert turn["state"] == "unclear"
        assert "repeated" in turn["error"]
        assert pipe.agent.questions == []

    def test_a_repetition_loop_is_caught_even_when_confident(self, rig) -> None:
        """The state must not depend on confidence: the field is what counts."""
        pipe, db, _ = rig
        pipe.stt.result = _transcript("ha ha ha", confidence=0.99, elapsed_ms=80, repetitive=True)
        turn = wait_for(db, submit_and_wait(pipe, db), TERMINAL)
        assert turn["state"] == "unclear"
        assert pipe.agent.questions == []

    def test_a_legitimate_repeated_digit_still_answers(self, rig) -> None:
        """"What is 2 + 2?" has two identical tokens in a row and is real."""
        pipe, db, _ = rig
        pipe.stt.result = _transcript("What is 2 + 2?", confidence=0.68, elapsed_ms=80)
        turn = wait_for(db, submit_and_wait(pipe, db), TERMINAL)
        assert turn["state"] == "done"
        assert "4" in turn["answer"]

    def test_audio_is_still_replayable_after_no_speech(self, rig) -> None:
        """A failed turn should still be audible, for debugging the room."""
        pipe, db, _ = rig
        pipe.stt.script = {"*": "um friday"}
        turn = wait_for(db, submit_and_wait(pipe, db), TERMINAL)
        assert turn["audio_path"] and Path(turn["audio_path"]).exists()


class TestFailures:
    def test_stt_crash_becomes_an_error_turn(self, rig) -> None:
        pipe, db, _ = rig
        pipe.stt.raise_on_next = RuntimeError("CUDA out of memory")
        turn = wait_for(db, submit_and_wait(pipe, db), TERMINAL)
        assert turn["state"] == "error"
        assert "transcription failed" in turn["error"]
        assert "CUDA out of memory" in turn["error"]

    def test_llm_crash_becomes_an_error_turn(self, rig) -> None:
        pipe, db, _ = rig
        pipe.stt.script = {"*": "tell me something"}
        pipe.agent.raise_on_next = RuntimeError("model unavailable")
        turn = wait_for(db, submit_and_wait(pipe, db), TERMINAL)
        assert turn["state"] == "error"
        assert "assistant failed" in turn["error"]
        # The transcript is still on record even though the answer failed.
        assert turn["transcript"] == "tell me something"

    def test_an_error_reply_without_text_becomes_an_error_state(self, rig) -> None:
        pipe, db, _ = rig
        pipe.stt.script = {"*": "tell me something"}
        pipe.agent.reply = AgentReply(text="", error="model unavailable")
        turn = wait_for(db, submit_and_wait(pipe, db), TERMINAL)
        assert turn["state"] == "error"
        assert turn["error"] == "model unavailable"

    def test_a_partial_reply_keeps_its_text(self, rig) -> None:
        pipe, db, _ = rig
        pipe.stt.script = {"*": "tell me something"}
        pipe.agent.reply = AgentReply(text="Sorry, I could not reach the weather service.", error="timeout")
        turn = wait_for(db, submit_and_wait(pipe, db), TERMINAL)
        assert turn["state"] == "done"
        assert "could not reach" in turn["answer"]


class TestTextTurns:
    def test_text_turn_needs_no_audio(self, rig) -> None:
        pipe, db, _ = rig
        turn = db.create_turn(source="manual")
        pipe.submit_text(turn["id"], "what is 144 divided by 12")
        done = wait_for(db, turn["id"], TERMINAL)
        assert done["state"] == "done"
        assert done["answer"] == "12."
        assert done["audio_path"] is None

    def test_queued_turns_are_processed_in_order(self, rig) -> None:
        pipe, db, _ = rig
        ids = []
        for n in (2, 3, 4):
            turn = db.create_turn(source="manual")
            pipe.submit_text(turn["id"], f"what is {n} times 2")
            ids.append(turn["id"])
        for turn_id in ids:
            turn = wait_for(db, turn_id, TERMINAL)
            assert turn["state"] == "done"
        assert [db.get_turn(i)["seq"] for i in ids] == sorted(db.get_turn(i)["seq"] for i in ids)


class TestRetention:
    def test_old_turns_are_pruned(self, rig) -> None:
        pipe, db, _ = rig
        pipe.cfg.retention.max_turns = 3
        for _ in range(6):
            turn = db.create_turn(source="manual")
            pipe.submit_text(turn["id"], "what is 1 times 1")
            wait_for(db, turn["id"], TERMINAL)
        assert db.count() <= 3 + 1  # pruning happens after the answer, so allow a race

    def test_pruned_audio_files_are_deleted(self, rig) -> None:
        pipe, db, _ = rig
        pipe.cfg.retention.max_turns = 2
        paths = []
        for i in range(5):
            path = pipe.cfg.recordings_dir / f"prune-{i}.wav"
            path.write_bytes(b"RIFF" + b"\x00" * 40)
            turn = db.create_turn(audio_path=str(path), duration_s=1.0)
            pipe.submit_text(turn["id"], "what is 2 times 2")
            paths.append(path)
            wait_for(db, turn["id"], TERMINAL)
        for path in paths[:-2]:
            assert not path.exists(), f"{path} was orphaned by pruning"

    def test_audio_can_be_kept_on_disk(self, rig) -> None:
        pipe, db, _ = rig
        pipe.cfg.retention.max_turns = 2
        pipe.cfg.retention.delete_audio = False
        paths = []
        for i in range(4):
            path = pipe.cfg.recordings_dir / f"keep-{i}.wav"
            path.write_bytes(b"RIFF" + b"\x00" * 40)
            turn = db.create_turn(audio_path=str(path), duration_s=1.0)
            pipe.submit_text(turn["id"], "what is 2 times 2")
            paths.append(path)
            wait_for(db, turn["id"], TERMINAL)
        assert all(p.exists() for p in paths[:-2])

    def test_retention_disabled_keeps_everything(self, rig) -> None:
        pipe, db, _ = rig
        pipe.cfg.retention.max_turns = 0
        for _ in range(4):
            turn = db.create_turn(source="manual")
            pipe.submit_text(turn["id"], "what is 1 times 1")
            wait_for(db, turn["id"], TERMINAL)
        assert db.count() == 4


class TestQueue:
    def test_queue_depth_is_visible(self, rig) -> None:
        pipe, _, _ = rig
        pipe.stop()  # nothing draining, so the depth must grow
        submit_audio(pipe)
        submit_audio(pipe)
        assert pipe.queue_depth == 2


class TestFormats:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"sample_rate": 16000, "channels": 1, "bits": 16},
            {"sample_rate": 44100, "channels": 1, "bits": 16},
            {"sample_rate": 48000, "channels": 2, "bits": 16},
            {"sample_rate": 16000, "channels": 1, "bits": 32},
        ],
    )
    def test_any_declared_format_is_recorded_and_reported(self, rig, kwargs) -> None:
        pipe, db, _ = rig
        turn = wait_for(db, submit_and_wait(pipe, db, seconds=0.2, **kwargs), TERMINAL)
        assert turn["sample_rate"] == kwargs["sample_rate"]
        assert turn["channels"] == kwargs["channels"]
        info = parse_wav_header(Path(turn["audio_path"]).read_bytes())
        assert info.sample_rate == kwargs["sample_rate"]
        assert info.channels == kwargs["channels"]

    def test_source_is_recorded(self, rig) -> None:
        pipe, db, _ = rig
        turn = wait_for(db, submit_and_wait(pipe, db, source="raw-vad"), TERMINAL)
        assert turn["source"] == "raw-vad"


def _transcript(text: str, *, confidence: float, elapsed_ms: int, repetitive: bool = False):
    from server.stt import Transcript

    return Transcript(
        text=text,
        confidence=confidence,
        duration_s=0.3,
        elapsed_ms=elapsed_ms,
        repetitive=repetitive,
    )
