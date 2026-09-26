"""End-to-end: a real audio fixture over a real socket to a real answer in the API.

This is the test that would have caught a broken contract between the firmware
and the server. Everything between is real -- the models, the TCP listener, the
worker thread, SQLite, the REST API -- so a regression anywhere in the chain
shows up as a wrong transcript or a missing answer rather than a stack trace in
a component that nobody wired together.
"""

from __future__ import annotations

import json
import socket
import time

import pytest
from fastapi.testclient import TestClient

from server.bus import EventBus
from server.config import Config
from server.db import Database
from server.pipeline import Pipeline
from server.web.api import create_app

from .conftest import FIXTURES, needs_llm, needs_stt

pytestmark = [needs_stt, needs_llm]


def _manifest() -> dict[str, dict]:
    return json.loads((FIXTURES / "manifest.json").read_text())


def _fixture_bytes(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def _send_over_tcp(port: int, payload: bytes, *, chunk: int = 4096) -> None:
    """Stream the way the firmware does: many small writes, then close.

    Closing is the end-of-utterance signal, so the framer has to cope with a
    header whose sizes are correct and a body that arrives in pieces.
    """
    with socket.create_connection(("127.0.0.1", port), timeout=10) as sock:
        for start in range(0, len(payload), chunk):
            sock.sendall(payload[start : start + chunk])
            time.sleep(0.001)  # deliberate: coalescing would hide framing bugs


def _wait_for_state(client: TestClient, turn_id: str, timeout: float = 90.0) -> dict:
    """Poll until the turn reaches a terminal state, as a browser would."""
    deadline = time.monotonic() + timeout
    turn: dict = {}
    while time.monotonic() < deadline:
        turn = client.get(f"/api/turns/{turn_id}").json()
        if turn["state"] in {"done", "no_speech", "unclear", "error"}:
            return turn
        time.sleep(0.25)
    raise AssertionError(f"turn {turn_id} never finished; last state was {turn.get('state')!r}")


@pytest.fixture
def live(cfg: Config):
    """A whole server on a real port: ingest listener, worker thread, HTTP API.

    Nothing is stubbed. The models come from ``models/``, the listener binds a
    real socket (port 0, so a leftover listener from another test cannot make
    this pass or fail for the wrong reason), and the pipeline runs on its own
    thread exactly as it does in production.
    """
    cfg.ingest.port = 0
    cfg.llm.eager_load = True

    db = Database(cfg.db_path)
    bus = EventBus()
    pipeline = Pipeline(cfg, db, bus)
    app = create_app(cfg, db=db, pipeline=pipeline, bus=bus, start_ingest=True)

    with TestClient(app) as client:
        yield client, app.state.ingest, db, pipeline
    db.close()


@pytest.mark.slow
# Every test here waits on real model inference and, for the weather case, two
# live HTTP calls; 5 minutes is a ceiling, not an expectation (the file runs in
# ~22s on a warm GPU).
@pytest.mark.timeout(300)
class TestEndToEnd:
    def test_math_fixture_gets_the_right_answer(self, live) -> None:
        client, ingest, db, _ = live
        _send_over_tcp(ingest.port, _fixture_bytes("q_math_7x23.wav"))

        turn = _wait_for_state(client, _only_turn(client))
        assert turn["state"] == "done", turn
        assert "7" in (turn["transcript"] or "")
        assert "23" in (turn["transcript"] or "")
        # 7 * 23 == 161
        assert "161" in (turn["answer"] or ""), turn
        assert turn["transcript_ms"] is not None
        assert [c["name"] for c in turn["tool_calls"]] == ["calculate"]

    def test_weather_fixture_calls_the_weather_tool(self, live) -> None:
        client, ingest, db, _ = live
        _send_over_tcp(ingest.port, _fixture_bytes("q_weather_pune.wav"))

        turn = _wait_for_state(client, _only_turn(client))
        assert turn["state"] == "done", turn
        # The hotword biasing exists so the city survives; the whole point of
        # this test is that "Pune" is not misheard as "Pooner".
        assert "pune" in (turn["transcript"] or "").lower(), turn
        names = [c["name"] for c in turn["tool_calls"]]
        assert names == ["get_weather"], turn
        args = turn["tool_calls"][0]["arguments"]
        assert "pune" in json.dumps(args).lower(), args
        assert "pune" in (turn["answer"] or "").lower(), turn

    def test_audio_is_persisted_and_playable(self, live) -> None:
        client, ingest, db, _ = live
        _send_over_tcp(ingest.port, _fixture_bytes("q_math_2_plus_2.wav"))
        turn = _wait_for_state(client, _only_turn(client))

        assert turn["audio_url"] == f"/api/audio/{turn['id']}"
        assert turn["sample_rate"] == 16_000
        assert turn["channels"] == 1
        assert 1.0 < turn["duration_s"] < 4.0

        # The browser fetches this URL; it must return a real WAV, because the
        # waveform decodes it with decodeAudioData.
        response = client.get(turn["audio_url"])
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
        assert response.content[:4] == b"RIFF"
        assert response.content[8:12] == b"WAVE"
        # Header (44) + data, so the RIFF size field must match the payload.
        assert int.from_bytes(response.content[4:8], "little") == len(response.content) - 8

    def test_a_typed_question_takes_the_same_path(self, live) -> None:
        client, _ingest, _db, _ = live
        created = client.post("/api/turns", json={"text": "what is 12 divided by 4"})
        assert created.status_code == 201

        turn = _wait_for_state(client, created.json()["id"])
        assert turn["state"] == "done", turn
        assert turn["transcript"] == "what is 12 divided by 4"
        assert "3" in (turn["answer"] or ""), turn

    def test_two_utterances_on_one_connection_are_two_turns(self, live) -> None:
        client, ingest, _db, _ = live
        first = _fixture_bytes("q_math_7x23.wav")
        second = _fixture_bytes("q_math_2_plus_2.wav")
        _send_over_tcp(ingest.port, first + second)

        deadline = time.monotonic() + 90.0
        turns: list[dict] = []
        while time.monotonic() < deadline:
            turns = client.get("/api/turns").json()
            if len(turns) == 2 and all(t["state"] in {"done", "error"} for t in turns):
                break
            time.sleep(0.25)
        assert len(turns) == 2, [t["state"] for t in turns]
        assert all(t["state"] == "done" for t in turns), turns
        # Each utterance got its own sequence number and its own audio file.
        assert {t["seq"] for t in turns} == {1, 2}
        assert len({t["audio_url"] for t in turns}) == 2


def _only_turn(client: TestClient) -> str:
    """Id of the single turn in the store, for a one-utterance test."""
    turns = client.get("/api/turns").json()
    assert len(turns) == 1, f"expected exactly one turn, got {len(turns)}"
    return turns[0]["id"]


@pytest.mark.timeout(600)  # sweeps all nine fixtures
def test_every_fixture_produces_a_terminal_turn(live) -> None:
    """Sweep the whole golden set: nothing may wedge, drop, or hang.

    Asserting a *terminal* state rather than a specific transcript is the point:
    this catches framer, STT and LLM failures without making the suite a
    transcript-accuracy benchmark (which ``tests/test_stt.py`` already is).
    """
    client, ingest, _db, _ = live
    manifest = _manifest()
    names = sorted(manifest)
    assert names, "the fixture manifest is empty"

    for name in names:
        _send_over_tcp(ingest.port, _fixture_bytes(name))

    deadline = time.monotonic() + 300.0
    turns: list[dict] = []
    while time.monotonic() < deadline:
        turns = client.get("/api/turns").json()
        if len(turns) == len(names) and all(
            t["state"] in {"done", "no_speech", "unclear", "error"} for t in turns
        ):
            break
        time.sleep(0.5)

    assert len(turns) == len(names), (
        f"only {len(turns)} of {len(names)} utterances became turns; "
        f"states={[t['state'] for t in turns]}"
    )
    stuck = [t for t in turns if t["state"] not in {"done", "no_speech", "unclear", "error"}]
    assert not stuck, stuck
    # Every utterance must have kept its audio, or the waveform has nothing to draw.
    assert all(t["audio_url"] for t in turns), [t["id"] for t in turns if not t["audio_url"]]
    # Silence and noise should be rejected, not answered.
    for turn in turns:
        if turn["state"] == "done":
            assert turn["answer"], turn
