"""SQLite persistence and the event bus.

These two are what let the UI show a turn progressing rather than appearing all
at once, so the state transitions and the fan-out both matter.
"""

from __future__ import annotations

import asyncio
import threading
from datetime import datetime, timedelta

import pytest

from server.bus import EventBus
from server.db import TERMINAL_STATES, Database, summarise


def make_turn(db: Database, **kwargs):
    return db.create_turn(**kwargs)


class TestCreate:
    def test_starts_as_uploaded(self, db: Database) -> None:
        turn = make_turn(db, audio_path="/tmp/a.wav", duration_s=1.5, sample_rate=16000, channels=1, source="wav")
        assert turn["state"] == "uploaded"
        assert turn["transcript"] is None
        assert turn["answer"] is None
        assert turn["tool_calls"] == []
        assert turn["seq"] == 1
        assert turn["id"]

    def test_seq_increments(self, db: Database) -> None:
        seqs = [make_turn(db)["seq"] for _ in range(3)]
        assert seqs == [1, 2, 3]

    def test_explicit_id_is_honoured(self, db: Database) -> None:
        turn = make_turn(db, id="abc123")
        assert turn["id"] == "abc123"
        assert db.get_turn("abc123") is not None

    def test_timestamps_are_iso_utc(self, db: Database) -> None:
        turn = make_turn(db)
        stamp = datetime.fromisoformat(turn["created_at"])
        assert stamp.tzinfo is not None
        assert stamp.utcoffset() == timedelta(0)


class TestUpdate:
    def test_patches_fields(self, db: Database) -> None:
        turn = make_turn(db)
        updated = db.update_turn(
            turn["id"],
            state="transcribing",
            transcript="what is 7 times 23",
            confidence=0.94,
            transcript_ms=101,
        )
        assert updated["state"] == "transcribing"
        assert updated["transcript"] == "what is 7 times 23"
        assert updated["confidence"] == pytest.approx(0.94)
        assert updated["transcript_ms"] == 101

    def test_none_does_not_clobber(self, db: Database) -> None:
        """Partial updates are the norm; a None must not erase a real value."""
        turn = make_turn(db)
        db.update_turn(turn["id"], transcript="hello", confidence=0.5)
        db.update_turn(turn["id"], state="done", transcript=None, confidence=None)
        after = db.get_turn(turn["id"])
        assert after["transcript"] == "hello"
        assert after["confidence"] == pytest.approx(0.5)
        assert after["state"] == "done"

    def test_tool_calls_are_serialised(self, db: Database) -> None:
        turn = make_turn(db)
        calls = [{"name": "calculate", "arguments": {"expression": "7 * 23"}, "duration_ms": 0}]
        db.update_turn(turn["id"], tool_calls=calls)
        assert db.get_turn(turn["id"])["tool_calls"] == calls

    def test_corrupt_tool_calls_json_degrades_to_empty(self, db: Database) -> None:
        turn = make_turn(db)
        db._connect().execute("UPDATE turns SET tool_calls = ? WHERE id = ?", ("{oops", turn["id"]))
        db._connect().commit()
        assert db.get_turn(turn["id"])["tool_calls"] == []

    def test_unknown_fields_are_ignored(self, db: Database) -> None:
        """A stray key must not become SQL, let alone be executed."""
        turn = make_turn(db)
        db.update_turn(turn["id"], nonsense="x", id="hacked")
        assert db.get_turn(turn["id"])["id"] == turn["id"]
        assert db.get_turn("hacked") is None

    def test_update_touches_updated_at(self, db: Database) -> None:
        turn = make_turn(db)
        updated = db.update_turn(turn["id"], state="done")
        assert updated["updated_at"] >= turn["updated_at"]

    def test_missing_turn_returns_none(self, db: Database) -> None:
        assert db.update_turn("nope", state="done") is None

    def test_empty_update_is_a_noop(self, db: Database) -> None:
        turn = make_turn(db)
        assert db.update_turn(turn["id"])["id"] == turn["id"]


class TestLifecycle:
    def test_full_pipeline_states(self, db: Database) -> None:
        """The exact sequence the UI renders, end to end."""
        turn = make_turn(db, audio_path="/tmp/t.wav", duration_s=2.0)
        for state, fields in [
            ("transcribing", {"transcript": "what is 7 times 23", "transcript_ms": 100}),
            ("thinking", {"confidence": 0.9}),
            ("done", {"answer": "161", "llm_ms": 950}),
        ]:
            got = db.update_turn(turn["id"], state=state, **fields)
            assert got["state"] == state
        final = db.get_turn(turn["id"])
        assert final["state"] in TERMINAL_STATES
        assert final["transcript"] == "what is 7 times 23"
        assert final["answer"] == "161"

    def test_error_state_keeps_the_transcript(self, db: Database) -> None:
        turn = make_turn(db)
        db.update_turn(turn["id"], state="transcribing", transcript="hello")
        db.update_turn(turn["id"], state="error", error="model unavailable")
        after = db.get_turn(turn["id"])
        assert after["error"] == "model unavailable"
        assert after["transcript"] == "hello"
        assert after["state"] == "error"

    def test_delete(self, db: Database) -> None:
        turn = make_turn(db)
        assert db.delete_turn(turn["id"]) is True
        assert db.get_turn(turn["id"]) is None
        assert db.delete_turn(turn["id"]) is False

    def test_count_and_latest(self, db: Database) -> None:
        assert db.count() == 0
        assert db.latest_turn() is None
        first = make_turn(db)
        make_turn(db)
        assert db.count() == 2
        assert db.latest_turn()["id"] != first["id"]


class TestListing:
    def test_newest_first(self, db: Database) -> None:
        ids = [make_turn(db)["id"] for _ in range(3)]
        listed = db.list_turns()
        assert [t["id"] for t in listed] == list(reversed(ids))

    def test_limit_and_offset(self, db: Database) -> None:
        for _ in range(5):
            make_turn(db)
        assert len(db.list_turns(limit=2)) == 2
        page2 = db.list_turns(limit=2, offset=2)
        assert len(page2) == 2
        assert page2[0]["id"] != db.list_turns(limit=2)[0]["id"]

    def test_all_audio_paths_skips_turns_without_audio(self, db: Database) -> None:
        make_turn(db, audio_path="/a.wav")
        make_turn(db)  # no_speech turn: nothing to replay
        make_turn(db, audio_path="/b.wav")
        assert set(db.all_audio_paths()) == {"/a.wav", "/b.wav"}


class TestPrune:
    def test_keeps_the_newest(self, db: Database) -> None:
        ids = [make_turn(db, audio_path=f"/{i}.wav")["id"] for i in range(5)]
        db.prune(max_turns=2)
        assert db.count() == 2
        assert {t["id"] for t in db.list_turns()} == set(ids[-2:])

    def test_reports_orphaned_audio(self, db: Database) -> None:
        for i in range(4):
            make_turn(db, audio_path=f"/{i}.wav")
        orphans = db.prune(max_turns=1, delete_audio=True)
        assert set(orphans) == {"/0.wav", "/1.wav", "/2.wav"}

    def test_can_keep_the_audio_files(self, db: Database) -> None:
        for i in range(4):
            make_turn(db, audio_path=f"/{i}.wav")
        orphans = db.prune(max_turns=1, delete_audio=False)
        assert set(orphans) == {"/0.wav", "/1.wav", "/2.wav"}

    def test_disabled_when_max_is_zero(self, db: Database) -> None:
        for _ in range(3):
            make_turn(db)
        assert db.prune(max_turns=0) == []
        assert db.count() == 3

    def test_noop_when_under_the_limit(self, db: Database) -> None:
        make_turn(db, audio_path="/a.wav")
        assert db.prune(max_turns=10) == []
        assert db.count() == 1


class TestPersistence:
    def test_survives_reopen(self, tmp_path) -> None:
        path = tmp_path / "friday.db"
        first = Database(path)
        turn = first.create_turn(audio_path="/a.wav", duration_s=1.0)
        first.update_turn(turn["id"], state="done", transcript="hi", answer="hello")
        first.close()

        second = Database(path)
        try:
            reloaded = second.get_turn(turn["id"])
            assert reloaded["transcript"] == "hi"
            assert reloaded["answer"] == "hello"
            assert reloaded["state"] == "done"
        finally:
            second.close()

    def test_schema_is_idempotent(self, tmp_path) -> None:
        path = tmp_path / "friday.db"
        Database(path).close()
        Database(path).close()  # must not raise on re-running CREATE TABLE

    def test_parent_directory_is_created(self, tmp_path) -> None:
        path = tmp_path / "deep" / "nested" / "friday.db"
        Database(path).close()
        assert path.exists()


class TestRecoverInterrupted:
    """A turn found mid-pipeline at startup belongs to a process that died."""

    def test_fails_every_non_terminal_turn(self, tmp_path) -> None:
        db = Database(tmp_path / "friday.db")
        live = db.create_turn()
        for state in ("transcribing", "thinking", "calling"):
            db.update_turn(live["id"], state=state)

        assert db.recover_interrupted() == [live["id"]]
        recovered = db.get_turn(live["id"])
        assert recovered["state"] == "error"
        assert "interrupted" in recovered["error"]

    def test_leaves_finished_turns_alone(self, tmp_path) -> None:
        db = Database(tmp_path / "friday.db")
        done = db.create_turn()
        spoke = db.create_turn()
        db.update_turn(done["id"], state="done", answer="42")
        db.update_turn(spoke["id"], state="no_speech")

        assert db.recover_interrupted() == []
        assert db.get_turn(done["id"])["answer"] == "42"
        assert db.get_turn(spoke["id"])["state"] == "no_speech"

    def test_is_idempotent(self, tmp_path) -> None:
        db = Database(tmp_path / "friday.db")
        db.create_turn()
        assert db.recover_interrupted()
        # Second call must find nothing left to fix, not re-error the row.
        assert db.recover_interrupted() == []

    def test_covers_exactly_the_inverted_terminal_set(self, tmp_path) -> None:
        """Guards against TERMINAL_STATES and the SQL list drifting apart."""
        db = Database(tmp_path / "friday.db")
        for state in sorted(TERMINAL_STATES):
            turn = db.create_turn()
            db.update_turn(turn["id"], state=state)
        assert db.recover_interrupted() == []

    def test_keeps_the_transcript_and_audio_of_an_interrupted_turn(self, tmp_path) -> None:
        """Failing the turn must not throw away what was already captured."""
        db = Database(tmp_path / "friday.db")
        turn = db.create_turn(audio_path="/tmp/a.wav", duration_s=1.5)
        db.update_turn(turn["id"], state="transcribing", transcript="what is seven")

        db.recover_interrupted()
        recovered = db.get_turn(turn["id"])
        assert recovered["transcript"] == "what is seven"
        assert recovered["audio_path"] == "/tmp/a.wav"
        assert recovered["duration_s"] == 1.5


class TestConcurrency:
    def test_writes_from_many_threads_do_not_lose_rows(self, tmp_path) -> None:
        """The pipeline writes from a worker thread while the API reads."""
        database = Database(tmp_path / "friday.db")
        errors: list[BaseException] = []

        def writer(n: int) -> None:
            try:
                for _ in range(10):
                    turn = database.create_turn(audio_path=f"/{n}.wav")
                    database.update_turn(turn["id"], state="done", answer=str(n))
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        try:
            assert not errors, errors
            assert database.count() == 80
            seqs = [t["seq"] for t in database.list_turns(limit=1000)]
            assert sorted(seqs) == list(range(1, 81))  # no duplicate or lost sequence
        finally:
            database.close()


class TestSummarise:
    def test_empty(self) -> None:
        assert summarise([]) == {
            "total": 0, "done": 0, "errors": 0,
            "avg_transcript_ms": None, "avg_llm_ms": None,
        }

    def test_averages_only_completed_turns(self) -> None:
        turns = [
            {"state": "done", "transcript_ms": 100, "llm_ms": 1000},
            {"state": "done", "transcript_ms": 200, "llm_ms": 2000},
            {"state": "transcribing", "transcript_ms": 999, "llm_ms": 999},
            {"state": "error"},
        ]
        out = summarise(turns)
        assert out["total"] == 4
        assert out["done"] == 2
        assert out["errors"] == 1
        assert out["avg_transcript_ms"] == 150
        assert out["avg_llm_ms"] == 1500

    def test_handles_missing_timings(self) -> None:
        assert summarise([{"state": "done"}])["avg_llm_ms"] is None


class TestBus:
    async def test_fans_out_to_every_subscriber(self, bus: EventBus) -> None:
        a, b = await bus.subscribe(), await bus.subscribe()
        await bus.publish({"type": "turn:updated", "id": "1"})
        assert (await a.get())["id"] == "1"
        assert (await b.get())["id"] == "1"

    async def test_unsubscribe_stops_delivery(self, bus: EventBus) -> None:
        q = await bus.subscribe()
        await bus.unsubscribe(q)
        await bus.publish({"type": "x"})
        assert q.empty()
        assert bus.subscriber_count == 0

    async def test_unsubscribe_is_idempotent(self, bus: EventBus) -> None:
        q = await bus.subscribe()
        await bus.unsubscribe(q)
        await bus.unsubscribe(q)

    async def test_a_stalled_subscriber_does_not_block_the_others(self, bus: EventBus) -> None:
        slow = await bus.subscribe()
        fast = await bus.subscribe()
        for i in range(300):  # more than the 256-slot queue
            await bus.publish({"i": i})
        assert not fast.empty() or slow.qsize() == 256
        assert slow.qsize() == 256  # the slow one filled up and was dropped, not awaited

    async def test_publish_before_binding_a_loop_is_a_noop(self) -> None:
        b = EventBus()
        b.publish_threadsafe({"type": "x"})  # must not raise

    async def test_publish_threadsafe_reaches_subscribers(self, bus: EventBus) -> None:
        loop = asyncio.get_running_loop()
        bus.bind_loop(loop)
        q = await bus.subscribe()
        bus.publish_threadsafe({"type": "from-thread"})
        # The coroutine is scheduled on the loop; give it a turn.
        await asyncio.sleep(0.05)
        assert (await q.get())["type"] == "from-thread"

    async def test_publish_threadsafe_from_a_real_thread(self, bus: EventBus) -> None:
        loop = asyncio.get_running_loop()
        bus.bind_loop(loop)
        q = await bus.subscribe()
        threading.Thread(target=lambda: bus.publish_threadsafe({"type": "t"})).start()
        await asyncio.sleep(0.1)
        assert not q.empty()
