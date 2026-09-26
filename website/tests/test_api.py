"""The HTTP surface and the WebSocket feed.

The pipeline is stubbed so these run instantly; the shapes asserted here are the
contract the frontend is written against.
"""

from __future__ import annotations

import base64
import json
import os
import queue
import socket
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from server.bus import EventBus
from server.config import Config
from server.db import Database
from server.pipeline import Pipeline
from server.web.api import create_app


class StubSTT:
    device = "cpu"
    compute_type = "int8"

    def __init__(self) -> None:
        self._loaded = True

    @property
    def loaded(self) -> bool:
        return self._loaded

    def transcribe(self, path: Path):
        raise AssertionError("no audio should reach the stub STT")


class StubAgent:
    def __init__(self) -> None:
        self._loaded = True
        self.questions: list[str] = []

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        pass

    def close(self) -> None:
        pass

    def answer(self, question: str):
        from server.agent import AgentReply

        self.questions.append(question)
        # Answer immediately on the calling thread so tests stay deterministic.
        return AgentReply(text=f"you asked: {question}", tool_calls=[])


@pytest.fixture
def app(cfg: Config):
    cfg.ingest.host = "127.0.0.1"
    cfg.ingest.port = 0
    db = Database(cfg.db_path)
    bus = EventBus()
    pipe = Pipeline(cfg, db, bus)
    pipe.stt = StubSTT()  # type: ignore[assignment]
    pipe.agent = StubAgent()  # type: ignore[assignment]
    # Answer on the caller's thread: no worker, no polling, no flakes.
    pipe._queue = _ImmediateQueue(pipe)  # type: ignore[assignment]
    application = create_app(cfg, db=db, pipeline=pipe, bus=bus, start_ingest=True)
    with TestClient(application) as client:
        client.app_state = (db, pipe)  # type: ignore[attr-defined]
        yield client
    db.close()


class _ImmediateQueue:
    """Runs each job the moment it arrives; parks the worker thread.

    The real queue is drained by a thread, which would make every assertion a
    race. This keeps the worker asleep and executes inline instead.
    """

    def __init__(self, pipe: Pipeline) -> None:
        self._pipe = pipe
        self._parked: queue.Queue[Any] = queue.Queue()

    def put(self, job) -> None:
        from server.pipeline import TextJob

        if job is None:
            self._parked.put(None)  # release the worker so stop() can join it
            return
        if isinstance(job, TextJob):
            self._pipe._answer(job.turn_id, job.text)
        else:
            self._pipe._process(job)

    def get(self):
        return self._parked.get()

    def qsize(self) -> int:
        return self._parked.qsize()

    def task_done(self) -> None:
        pass


# ------------------------------------------------------------------- health


class TestHealth:
    def test_reports_every_subsystem(self, app: TestClient) -> None:
        body = app.get("/api/health").json()
        assert body["status"] in {"ok", "degraded"}
        assert body["ingest"] == "listening"
        assert body["stt"] == "ready"
        assert body["llm"] == "ready"
        assert body["turns"] == 0
        assert body["device_connected"] is False

    def test_degraded_when_ingest_is_down(self, cfg: Config) -> None:
        db = Database(cfg.db_path)
        try:
            pipe = Pipeline(cfg, db, EventBus())
            pipe.stt = StubSTT()  # type: ignore[assignment]
            pipe.agent = StubAgent()  # type: ignore[assignment]
            app = create_app(cfg, db=db, pipeline=pipe, start_ingest=False)
            with TestClient(app) as client:
                body = client.get("/api/health").json()
                assert body["status"] == "degraded"
                assert body["ingest"] == "down"
        finally:
            db.close()

    def test_ingest_stats(self, app: TestClient) -> None:
        body = app.get("/api/ingest/stats").json()
        assert body["listening"] is True
        assert body["port"] > 0
        assert body["queue_depth"] == 0
        assert body["bytes_received"] == 0
        assert body["stt_compute_type"] == "int8"

    def test_system_endpoint(self, app: TestClient) -> None:
        body = app.get("/api/system").json()
        assert "cpu" in json.dumps(body).lower() or "memory" in json.dumps(body).lower()


class TestStartupRecovery:
    """A turn left mid-pipeline by a dead process must not spin forever."""

    def test_an_interrupted_turn_is_failed_on_boot(self, cfg: Config) -> None:
        db = Database(cfg.db_path)
        try:
            stranded = db.create_turn(audio_path=None, duration_s=1.2)
            db.update_turn(stranded["id"], state="transcribing")
            finished = db.create_turn()
            db.update_turn(finished["id"], state="done", answer="161")

            pipe = Pipeline(cfg, db, EventBus())
            pipe.stt = StubSTT()  # type: ignore[assignment]
            pipe.agent = StubAgent()  # type: ignore[assignment]
            app = create_app(cfg, db=db, pipeline=pipe, start_ingest=False)
            with TestClient(app) as client:
                recovered = client.get(f"/api/turns/{stranded['id']}").json()
                assert recovered["state"] == "error"
                assert "interrupted" in recovered["error"]
                # The finished turn is untouched.
                assert client.get(f"/api/turns/{finished['id']}").json()["answer"] == "161"
        finally:
            db.close()

    def test_a_stranded_turn_does_not_block_new_work(self, cfg: Config) -> None:
        """An old stuck turn must not hold up or confuse a fresh one."""
        db = Database(cfg.db_path)
        try:
            stranded = db.create_turn()
            db.update_turn(stranded["id"], state="thinking")

            pipe = Pipeline(cfg, db, EventBus())
            pipe.stt = StubSTT()  # type: ignore[assignment]
            pipe.agent = StubAgent()  # type: ignore[assignment]
            app = create_app(cfg, db=db, pipeline=pipe, start_ingest=False)
            with TestClient(app) as client:
                created = client.post("/api/turns", json={"text": "hello"})
                assert created.status_code == 201
                new_id = created.json()["id"]

                # The stranded turn is failed, the new one is left to the worker.
                assert client.get(f"/api/turns/{stranded['id']}").json()["state"] == "error"
                assert client.get(f"/api/turns/{new_id}").json()["state"] != "error"

                # Both are listed, and the newcomer is the newest row.
                listed = client.get("/api/turns").json()
                assert {t["id"] for t in listed} == {stranded["id"], new_id}
                assert listed[0]["id"] == new_id
        finally:
            db.close()


# -------------------------------------------------------------------- turns


class TestTurnList:
    def test_empty(self, app: TestClient) -> None:
        assert app.get("/api/turns").json() == []

    def test_newest_first(self, app: TestClient) -> None:
        for text in ("what is 7 times 23", "what time is it"):
            app.post("/api/turns", json={"text": text})
        ids = [t["id"] for t in app.get("/api/turns").json()]
        assert len(ids) == 2
        seqs = [t["seq"] for t in app.get("/api/turns").json()]
        assert seqs == sorted(seqs, reverse=True)

    def test_limit_is_clamped(self, app: TestClient) -> None:
        assert app.get("/api/turns?limit=0").status_code == 200
        assert app.get("/api/turns?limit=99999").status_code == 200
        assert app.get("/api/turns?limit=-5").status_code == 200

    def test_pagination(self, app: TestClient) -> None:
        for i in range(5):
            app.post("/api/turns", json={"text": f"question {i}"})
        page1 = app.get("/api/turns?limit=2").json()
        page2 = app.get("/api/turns?limit=2&offset=2").json()
        assert len(page1) == len(page2) == 2
        assert not {t["id"] for t in page1} & {t["id"] for t in page2}


class TestManualTurn:
    def test_a_typed_question_is_answered(self, app: TestClient) -> None:
        resp = app.post("/api/turns", json={"text": "what is 7 times 23"})
        assert resp.status_code == 201
        created = resp.json()
        assert created["state"] == "uploaded"
        assert created["source"] == "manual"

        done = app.get(f"/api/turns/{created['id']}").json()
        assert done["state"] == "done"
        # The typed text is kept verbatim: the transcript column is what the user
        # said or typed, and rewriting their wording would be misleading.
        assert done["transcript"] == "what is 7 times 23"
        assert done["answer"] == "161."

    def test_empty_text_is_rejected(self, app: TestClient) -> None:
        assert app.post("/api/turns", json={"text": ""}).status_code == 422

    def test_missing_field_is_rejected(self, app: TestClient) -> None:
        assert app.post("/api/turns", json={}).status_code == 422

    def test_absurdly_long_text_is_rejected(self, app: TestClient) -> None:
        assert app.post("/api/turns", json={"text": "x" * 5000}).status_code == 422

    def test_arithmetic_needs_no_llm(self, app: TestClient) -> None:
        """The fast path must be visible through the API, not just internally."""
        resp = app.post("/api/turns", json={"text": "what is 144 divided by 12"})
        turn = app.get(f"/api/turns/{resp.json()['id']}").json()
        assert turn["answer"] == "12."
        assert turn["llm_ms"] == 0
        assert turn["tool_calls"][0]["name"] == "calculate"

    def test_unicode_is_preserved(self, app: TestClient) -> None:
        resp = app.post("/api/turns", json={"text": "what is 7 times 23 你好 🎤"})
        turn = app.get(f"/api/turns/{resp.json()['id']}").json()
        assert "你好" in (turn["transcript"] or "")


class TestSingleTurn:
    def test_404_for_unknown_id(self, app: TestClient) -> None:
        assert app.get("/api/turns/nope").status_code == 404

    def test_shape_is_stable(self, app: TestClient) -> None:
        """The frontend depends on these keys existing, even when null."""
        resp = app.post("/api/turns", json={"text": "what is 7 times 23"})
        body = app.get(f"/api/turns/{resp.json()['id']}").json()
        for key in (
            "id", "seq", "created_at", "updated_at", "state", "audio_url",
            "duration_s", "sample_rate", "channels", "source", "transcript",
            "confidence", "answer", "error", "tool_calls", "transcript_ms", "llm_ms",
        ):
            assert key in body, key
        assert isinstance(body["tool_calls"], list)
        assert body["audio_url"] is None  # a typed turn has no audio

    def test_audio_url_is_set_when_theres_audio(self, app: TestClient) -> None:
        db, pipe = app.app_state  # type: ignore[attr-defined]
        turn = db.create_turn(audio_path="/tmp/x.wav", duration_s=1.0)
        body = app.get(f"/api/turns/{turn['id']}").json()
        assert body["audio_url"] == f"/api/audio/{turn['id']}"


class TestDelete:
    def test_removes_the_turn_and_its_audio(self, app: TestClient) -> None:
        db, _ = app.app_state  # type: ignore[attr-defined]
        path = Path(app.app_state[1].cfg.recordings_dir) / "delete-me.wav"  # type: ignore[attr-defined]
        path.write_bytes(b"RIFF" + b"\x00" * 40)
        turn = db.create_turn(audio_path=str(path), duration_s=1.0)

        assert app.delete(f"/api/turns/{turn['id']}").status_code == 204
        assert app.get(f"/api/turns/{turn['id']}").status_code == 404
        assert not path.exists()

    def test_404_for_unknown_id(self, app: TestClient) -> None:
        assert app.delete("/api/turns/nope").status_code == 404

    def test_deleting_a_text_turn_works(self, app: TestClient) -> None:
        resp = app.post("/api/turns", json={"text": "what is 2 plus 2"})
        assert app.delete(f"/api/turns/{resp.json()['id']}").status_code == 204


class TestAudio:
    def test_serves_the_recording(self, app: TestClient) -> None:
        from .conftest import tone_pcm, wav_bytes

        db, pipe = app.app_state  # type: ignore[attr-defined]
        pcm = tone_pcm(0.5)
        path = pipe.cfg.recordings_dir / "serve.wav"
        path.write_bytes(wav_bytes(pcm))
        turn = db.create_turn(audio_path=str(path), duration_s=0.5)

        resp = app.get(f"/api/audio/{turn['id']}")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "audio/wav"
        assert resp.content == path.read_bytes()

    def test_404_when_there_is_no_audio(self, app: TestClient) -> None:
        resp = app.post("/api/turns", json={"text": "what is 2 plus 2"})
        assert app.get(f"/api/audio/{resp.json()['id']}").status_code == 404

    def test_404_when_the_file_has_been_deleted(self, app: TestClient) -> None:
        db, _ = app.app_state  # type: ignore[attr-defined]
        turn = db.create_turn(audio_path="/tmp/definitely-not-here.wav")
        assert app.get(f"/api/audio/{turn['id']}").status_code == 404

    def test_404_for_unknown_id(self, app: TestClient) -> None:
        assert app.get("/api/audio/nope").status_code == 404


class TestStats:
    def test_aggregates(self, app: TestClient) -> None:
        app.post("/api/turns", json={"text": "what is 7 times 23"})
        app.post("/api/turns", json={"text": "what is the weather in Pune"})
        body = app.get("/api/stats").json()
        assert body["total"] == 2
        assert body["done"] == 2
        assert body["errors"] == 0

    def test_empty(self, app: TestClient) -> None:
        assert app.get("/api/stats").json()["total"] == 0


# ---------------------------------------------------------------- websocket


class TestWebSocket:
    def test_replays_history_on_connect(self, app: TestClient) -> None:
        app.post("/api/turns", json={"text": "what is 7 times 23"})
        with app.websocket_connect("/ws") as ws:
            replay = [ws.receive_json() for _ in range(1)]
            assert replay[0]["event"] == "turn.created"
            assert replay[0]["turn"]["answer"] == "161."

    def test_receives_live_updates(self, app: TestClient) -> None:
        """A client watching while a turn runs must see it progress."""
        with app.websocket_connect("/ws") as ws:
            app.post("/api/turns", json={"text": "what is 7 times 23"})
            states: list[str] = []
            for _ in range(6):
                msg = ws.receive_json()
                if msg.get("turn", {}).get("state") == "done":
                    states.append("done")
                    break
                states.append(msg["turn"]["state"])
            assert states[-1] == "done"
            assert "transcribing" in states or "thinking" in states

    def test_events_carry_the_public_turn_shape(self, app: TestClient) -> None:
        with app.websocket_connect("/ws") as ws:
            app.post("/api/turns", json={"text": "what is 7 times 23"})
            msg = ws.receive_json()
            assert "audio_path" not in msg["turn"]
            assert set(msg) >= {"event", "turn"}

    def test_two_clients_both_see_events(self, app: TestClient) -> None:
        with app.websocket_connect("/ws") as a, app.websocket_connect("/ws") as b:
            app.post("/api/turns", json={"text": "what is 7 times 23"})
            for ws in (a, b):
                seen = []
                for _ in range(6):
                    msg = ws.receive_json()
                    seen.append(msg["turn"]["state"])
                    if seen[-1] == "done":
                        break
                assert seen[-1] == "done"

    def test_disconnect_unsubscribes(self, app: TestClient) -> None:
        bus: EventBus = app.app.state.bus
        assert bus.subscriber_count == 0
        with app.websocket_connect("/ws"):
            assert bus.subscriber_count == 1
        # The server needs a moment to notice the close frame.
        for _ in range(50):
            if bus.subscriber_count == 0:
                break
            import time

            time.sleep(0.02)
        assert bus.subscriber_count == 0


class TestFrontendFallback:
    def test_root_explains_how_to_build_the_frontend(self, app: TestClient) -> None:
        """Before `npm run build` there is no dist; say so helpfully."""
        if Path("web/dist/assets").exists():
            assert app.get("/").status_code == 200
        else:
            body = app.get("/").json()
            assert "npm run build" in body["hint"]
            assert "/api/turns" in body["api"]


class TestGracefulShutdown:
    """Ctrl+C has to work while the console is open.

    The WebSocket feed is a long-lived task, and uvicorn waits for in-flight
    tasks before exiting. Without an explicit shutdown signal the server sits at
    "Waiting for background tasks to complete" forever and the operator has to
    signal it a second time -- which, in practice, means an orphaned process
    still holding the ingest port and the GPU.
    """

    @staticmethod
    def _open_websocket(port: int) -> socket.socket:
        """Complete a WebSocket handshake and deliberately leave it open."""
        key = base64.b64encode(os.urandom(16)).decode()
        sock = socket.create_connection(("127.0.0.1", port), timeout=10)
        sock.sendall(
            f"GET /ws HTTP/1.1\r\n"
            f"Host: 127.0.0.1:{port}\r\n"
            f"Upgrade: websocket\r\n"
            f"Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            f"Sec-WebSocket-Version: 13\r\n\r\n".encode()
        )
        status = sock.recv(200).split(b"\r\n", 1)[0]
        assert b"101" in status, status
        return sock

    def test_shutdown_completes_with_a_websocket_connected(self, cfg: Config) -> None:
        pytest.importorskip("uvicorn")
        import uvicorn

        server = uvicorn.Server(
            uvicorn.Config(
                create_app(cfg, start_ingest=False),
                host="127.0.0.1",
                port=0,
                log_level="warning",
            )
        )
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()

        deadline = time.monotonic() + 30
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.05)
        assert server.started, "uvicorn did not come up"
        port = server.servers[0].sockets[0].getsockname()[1]

        ws = self._open_websocket(port)
        try:
            # Give the handler a moment to reach its wait on the event queue,
            # which is the state that used to hang shutdown.
            time.sleep(0.5)
            server.should_exit = True
            thread.join(timeout=20)
            assert not thread.is_alive(), (
                "server did not exit with a websocket connected; the feed must "
                "also receive, or it never notices the socket closing"
            )
        finally:
            ws.close()
            server.should_exit = True
            thread.join(timeout=10)
