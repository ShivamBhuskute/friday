#!/usr/bin/env python
"""Smoke test: does the whole thing work, on this machine, right now?

Answers the question you actually have before a demo -- "if I speak to it, will
something come back?" -- without a browser and without the ESP32. Each check is
independent and reports pass/fail; the exit code is the number of failures.

    .venv/bin/python scripts/smoke.py

Options:
    --keep-going     run every check even after one fails (the default)
    --fast           skip the checks that need the LLM (saves ~30s)
    --json           emit machine-readable results
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from array import array
from dataclasses import asdict, dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FIXTURES = PROJECT_ROOT / "fixtures"

from server.ingest import protocol  # noqa: E402
from server.ingest.wav import parse_wav_header  # noqa: E402

GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""
    skipped: bool = False
    seconds: float = 0.0
    data: dict = field(default_factory=dict)


def _colour(ok: bool, skipped: bool = False) -> str:
    if skipped:
        return YELLOW
    return GREEN if ok else RED


class Server:
    """A real server in a subprocess, so this tests what a user would run."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.proc: subprocess.Popen | None = None
        self.api_port = 0
        self.ingest_port = 0

    def start(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        # Port 0 lets the OS pick, so two smoke runs cannot collide. The env
        # override scheme is FRIDAY_SECTION__KEY (see server/config.py).
        env = {
            **os.environ,
            "FRIDAY_DATA__DIR": str(self.data_dir),
            "FRIDAY_SERVER__PORT": "0",
            "FRIDAY_INGEST__PORT": "0",
        }
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "server"],
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self.api_port, self.ingest_port = _wait_for_ports(self.proc)

    def stop(self) -> None:
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.proc = None
        # The point of a smoke run is to leave nothing behind: a stale
        # /tmp/friday-smoke-<pid> per invocation is how /tmp quietly fills up.
        shutil.rmtree(self.data_dir, ignore_errors=True)


def _wait_for_ports(proc: subprocess.Popen) -> tuple[int, int]:
    """Read the bound ports from the server's own log lines.

    The config uses port 5000/8000; the probe asks for ephemeral ones so two
    smoke runs cannot collide. The server logs what it actually bound, which is
    more reliable than assuming 0 resolved the way we expect.
    """
    deadline = time.monotonic() + 120
    api = ingest = 0
    buffer = ""
    assert proc.stdout is not None
    while time.monotonic() < deadline:
        line = proc.stdout.readline()
        if not line:
            if proc.poll() is not None:
                raise RuntimeError("the server exited before it started listening")
            continue
        buffer += line
        match = re.search(r"Uvicorn running on http://[\d.]+:(\d+)", line)
        if match:
            api = int(match.group(1))
        match = re.search(r"ingest listening on [\d.]+:(\d+)", line)
        if match:
            ingest = int(match.group(1))
        if api and ingest:
            return api, ingest
    raise RuntimeError(f"the server never reported its ports; log so far:\n{buffer[-2000:]}")


# --------------------------------------------------------------------- checks


def check_imports() -> Check:
    t0 = time.monotonic()
    try:
        import faster_whisper  # noqa: F401
        import llama_cpp  # noqa: F401
        import numpy  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return Check("imports", False, f"{type(exc).__name__}: {exc}", seconds=time.monotonic() - t0)
    return Check("imports", True, "faster-whisper, llama-cpp, numpy", seconds=time.monotonic() - t0)


def check_models() -> Check:
    t0 = time.monotonic()
    stt = PROJECT_ROOT / "models" / "faster-whisper-small.en" / "model.bin"
    llm = PROJECT_ROOT / "models" / "qwen2.5-3b-instruct-q4_k_m.gguf"
    missing = [p.name for p in (stt, llm) if not p.exists()]
    if missing:
        return Check(
            "model weights",
            False,
            f"missing {', '.join(missing)} -- run scripts/fetch_models.py",
            seconds=time.monotonic() - t0,
        )
    size_mb = (stt.stat().st_size + llm.stat().st_size) / 1e6
    return Check(
        "model weights",
        True,
        f"present ({size_mb:.0f} MB)",
        seconds=time.monotonic() - t0,
    )


def check_gpu() -> Check:
    t0 = time.monotonic()
    try:
        from server.stt import _cuda_available
    except Exception as exc:  # noqa: BLE001
        return Check("gpu", False, f"{type(exc).__name__}: {exc}", seconds=time.monotonic() - t0)
    if _cuda_available():
        return Check("gpu", True, "visible to CTranslate2", seconds=time.monotonic() - t0)
    return Check(
        "gpu",
        True,
        "not visible -- STT will run on CPU (slow but correct)",
        seconds=time.monotonic() - t0,
        data={"cuda": False},
    )


def check_health(server: Server) -> Check:
    t0 = time.monotonic()
    body = _get_json(server.api_port, "/api/health")
    if body is None:
        return Check("server health", False, "no response", seconds=time.monotonic() - t0)
    ok = body.get("status") in {"ok", "degraded"}
    detail = f"status={body.get('status')} stt={body.get('stt')} llm={body.get('llm')}"
    if body.get("ingest") != "listening":
        return Check("server health", False, detail, seconds=time.monotonic() - t0)
    return Check("server health", ok, detail, seconds=time.monotonic() - t0, data=body)


def check_frontend(server: Server) -> Check:
    t0 = time.monotonic()
    status, body, _ = _get(server.api_port, "/")
    if status != 200 or b"<div id=\"root\"" not in body:
        return Check(
            "frontend served",
            False,
            "GET / did not return the built console -- run: cd web && npm run build",
            seconds=time.monotonic() - t0,
        )
    return Check("frontend served", True, "SPA is mounted at /", seconds=time.monotonic() - t0)


def _fixture_pcm(wav: Path) -> bytes:
    """The samples out of a fixture, with the RIFF header removed.

    The device does not send a header, so neither do we: the whole point of
    these checks is to exercise the path the ESP32 actually takes.
    """
    raw = wav.read_bytes()
    info = parse_wav_header(raw)
    return raw[info.data_offset : info.data_offset + info.data_size]


def _stream_and_wait(
    server: Server, wav: Path, timeout: float = 120.0, *, pcm: bytes | None = None
) -> dict:
    """Send a fixture over TCP exactly as the firmware does, then wait for the answer.

    Headerless PCM16, one 3840-byte hop at a time, truncated at the device's own
    silence gate, then the socket closed to signal the end of the utterance.
    """
    samples = _fixture_pcm(wav) if pcm is None else pcm
    payload = b"".join(protocol.firmware_hops(samples))
    before = _turn_ids(server.api_port)
    with socket.create_connection(("127.0.0.1", server.ingest_port), timeout=10) as sock:
        for start in range(0, len(payload), protocol.BYTES_PER_HOP):
            sock.sendall(payload[start : start + protocol.BYTES_PER_HOP])
            time.sleep(0.001)
    # Closing the socket is the end-of-utterance signal.

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        new = [t for t in _get_json(server.api_port, "/api/turns?limit=10") or [] if t["id"] not in before]
        if new and new[0]["state"] in {"done", "no_speech", "unclear", "error"}:
            return new[0]
        time.sleep(0.3)
    raise TimeoutError("the turn never reached a terminal state")


def _new_turns(server: Server, before: set[str], settle: float = 6.0) -> list[dict]:
    """Every turn that showed up, after giving a second one time to appear.

    A single question must produce a single turn. If the server's silence gate is
    stricter than the device's, one question with a natural pause in it arrives as
    two turns, and the user gets two answers to one question.
    """
    deadline = time.monotonic() + settle
    seen: dict[str, dict] = {}
    while time.monotonic() < deadline:
        for turn in _get_json(server.api_port, "/api/turns?limit=20") or []:
            if turn["id"] not in before:
                seen[turn["id"]] = turn
        time.sleep(0.4)
    return list(seen.values())


def _turn_ids(port: int) -> set[str]:
    return {t["id"] for t in (_get_json(port, "/api/turns?limit=200") or [])}


def check_mid_sentence_pause_stays_one_turn(server: Server) -> Check:
    """One question, one turn -- even with a natural pause in the middle.

    This is the check that the device's silence gate and the server's have to
    agree. Real speech pauses; a server that treats the pause as end-of-turn
    answers half a question, then answers the rest as if it were a new one. The
    fixtures are single clean phrases with no internal pause, so nothing else
    here would catch it.
    """
    t0 = time.monotonic()
    wav = FIXTURES / "q_weather_pune.wav"
    if not wav.exists():
        return Check("mid-sentence pause is one turn", False, f"missing {wav.name}", skipped=True)

    # A quiet stretch at 0.010: above the device's 0.007 gate, so the device
    # keeps streaming, but below a naively-chosen 0.020 server gate.
    gap_samples = int(protocol.SAMPLE_RATE * 1.0)
    gap = array("h", (int(0.010 * 32768) for _ in range(gap_samples))).tobytes()
    try:
        samples = _fixture_pcm(wav)
    except Exception as exc:  # noqa: BLE001
        return Check("mid-sentence pause is one turn", False, f"{type(exc).__name__}: {exc}", skipped=True)
    cut = len(samples) // 2 & ~(protocol.FRAME_BYTES - 1)
    pcm = samples[:cut] + gap + samples[cut:]

    before = _turn_ids(server.api_port)
    try:
        _stream_and_wait(server, wav, pcm=pcm)
    except Exception as exc:  # noqa: BLE001
        return Check(
            "mid-sentence pause is one turn", False, f"{type(exc).__name__}: {exc}",
            seconds=time.monotonic() - t0,
        )

    turns = _new_turns(server, before)
    elapsed = time.monotonic() - t0
    if len(turns) != 1:
        detail = "; ".join(
            f"{(t.get('transcript') or t['state'])!r} {t.get('duration_s', 0):.2f}s" for t in turns
        )
        return Check(
            "mid-sentence pause is one turn", False,
            f"expected 1 turn, got {len(turns)} -- the server's vad_rms_threshold "
            f"is stricter than the device's STREAM_SILENCE_RMS_GATE "
            f"({protocol.SILENCE_RMS_GATE}). Got: {detail}",
            seconds=elapsed, data={"turns": turns},
        )
    return Check(
        "mid-sentence pause is one turn", True,
        f"one turn, {turns[0].get('duration_s', 0):.2f}s", seconds=elapsed,
    )


def check_math(server: Server) -> Check:
    """The whole chain, with the fastest possible answer path."""
    t0 = time.monotonic()
    wav = FIXTURES / "q_math_7x23.wav"
    if not wav.exists():
        return Check("speech -> answer (math)", False, f"missing fixture {wav.name}", skipped=True)
    try:
        turn = _stream_and_wait(server, wav)
    except Exception as exc:  # noqa: BLE001
        return Check(
            "speech -> answer (math)", False, f"{type(exc).__name__}: {exc}",
            seconds=time.monotonic() - t0,
        )
    elapsed = time.monotonic() - t0
    if turn["state"] != "done":
        return Check(
            "speech -> answer (math)", False, f"state={turn['state']} {turn.get('error') or ''}",
            seconds=elapsed, data=turn,
        )
    if "161" not in (turn["answer"] or ""):
        return Check(
            "speech -> answer (math)", False,
            f"heard {turn['transcript']!r}, answered {turn['answer']!r}, expected 161",
            seconds=elapsed, data=turn,
        )
    return Check(
        "speech -> answer (math)", True,
        f"{turn['transcript']!r} -> {turn['answer']!r} in {elapsed:.1f}s "
        f"(stt {turn['transcript_ms']}ms, llm {turn['llm_ms']}ms)",
        seconds=elapsed, data=turn,
    )


def check_weather(server: Server) -> Check:
    """Needs the network, and is the slowest path: two HTTP calls to Open-Meteo."""
    t0 = time.monotonic()
    wav = FIXTURES / "q_weather_pune.wav"
    if not wav.exists():
        return Check("speech -> tool call (weather)", False, f"missing {wav.name}", skipped=True)
    try:
        turn = _stream_and_wait(server, wav, timeout=180.0)
    except Exception as exc:  # noqa: BLE001
        return Check(
            "speech -> tool call (weather)", False, f"{type(exc).__name__}: {exc}",
            seconds=time.monotonic() - t0,
        )
    elapsed = time.monotonic() - t0
    tools = [c["name"] for c in turn["tool_calls"]]
    if turn["state"] != "done":
        return Check(
            "speech -> tool call (weather)", False, f"state={turn['state']}",
            seconds=elapsed, data=turn,
        )
    if "get_weather" not in tools:
        return Check(
            "speech -> tool call (weather)", False,
            f"no weather call; tools={tools}, answer={turn['answer']!r}",
            seconds=elapsed, data=turn,
        )
    if "pune" not in (turn["answer"] or "").lower():
        return Check(
            "speech -> tool call (weather)", False,
            f"the answer lost the city: {turn['answer']!r} (transcript {turn['transcript']!r})",
            seconds=elapsed, data=turn,
        )
    return Check(
        "speech -> tool call (weather)", True,
        f"{turn['transcript']!r} -> {turn['answer']!r} in {elapsed:.1f}s",
        seconds=elapsed, data=turn,
    )


def check_audio_playback(server: Server) -> Check:
    """The waveform depends on this: the served bytes must be a real WAV."""
    t0 = time.monotonic()
    turns = _get_json(server.api_port, "/api/turns?limit=1") or []
    if not turns or not turns[0].get("audio_url"):
        return Check("audio playback", False, "no turn has audio", seconds=time.monotonic() - t0)
    status, body, ctype = _get(server.api_port, turns[0]["audio_url"])
    if status != 200:
        return Check("audio playback", False, f"HTTP {status}", seconds=time.monotonic() - t0)
    if body[:4] != b"RIFF" or body[8:12] != b"WAVE":
        return Check("audio playback", False, "not a RIFF/WAVE payload", seconds=time.monotonic() - t0)
    if "audio/wav" not in ctype:
        return Check("audio playback", False, f"content-type is {ctype}", seconds=time.monotonic() - t0)
    # A truncated RIFF size is the classic streaming bug; the browser refuses it.
    declared = int.from_bytes(body[4:8], "little")
    if declared != len(body) - 8:
        return Check(
            "audio playback", False,
            f"RIFF size says {declared} but the body is {len(body) - 8} bytes",
            seconds=time.monotonic() - t0,
        )
    return Check(
        "audio playback", True, f"{len(body)} bytes of valid WAV", seconds=time.monotonic() - t0
    )


def check_typed_turn(server: Server) -> Check:
    """The console must work with no ESP32 attached."""
    t0 = time.monotonic()
    body = json.dumps({"text": "what is 7 times 23"}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{server.api_port}/api/turns",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            created = json.loads(response.read())
    except Exception as exc:  # noqa: BLE001
        return Check(
            "typed turn (no hardware)", False, f"{type(exc).__name__}: {exc}",
            seconds=time.monotonic() - t0,
        )
    turn = _stream_and_wait_for(server, created["id"], timeout=60.0)
    ok = turn["state"] == "done" and "161" in (turn["answer"] or "")
    return Check(
        "typed turn (no hardware)", ok,
        f"{turn['transcript']!r} -> {turn['answer']!r}" if ok else f"{turn['state']}: {turn.get('error')}",
        seconds=time.monotonic() - t0, data=turn,
    )


def _stream_and_wait_for(server: Server, turn_id: str, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        turn = _get_json(server.api_port, f"/api/turns/{turn_id}")
        if turn and turn["state"] in {"done", "no_speech", "unclear", "error"}:
            return turn
        time.sleep(0.25)
    raise TimeoutError(f"turn {turn_id} never finished")


# ----------------------------------------------------------------- transport


def _get(port: int, path: str) -> tuple[int, bytes, str]:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=15) as response:
            return response.status, response.read(), response.headers.get("content-type", "")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), ""
    except Exception:  # noqa: BLE001
        return 0, b"", ""


def _get_json(port: int, path: str) -> dict | list | None:
    status, body, _ = _get(port, path)
    if status != 200:
        return None
    try:
        return json.loads(body)
    except ValueError:
        return None


# ---------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fast", action="store_true", help="skip the LLM-dependent checks")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    results: list[Check] = []

    if not args.json:
        print(f"{BOLD}FRIDAY smoke test{RESET} {DIM}{PROJECT_ROOT}{RESET}\n")

    results.append(check_imports())
    results.append(check_models())
    results.append(check_gpu())

    server = Server(Path("/tmp") / f"friday-smoke-{os.getpid()}")
    try:
        server.start()
        results.append(check_health(server))
        results.append(check_frontend(server))
        results.append(check_math(server))
        results.append(check_mid_sentence_pause_stays_one_turn(server))
        if not args.fast:
            results.append(check_weather(server))
        else:
            results.append(
                Check("speech -> tool call (weather)", True, "skipped (--fast)", skipped=True)
            )
        results.append(check_audio_playback(server))
        results.append(check_typed_turn(server))
    except Exception as exc:  # noqa: BLE001
        results.append(Check("server", False, f"{type(exc).__name__}: {exc}"))
    finally:
        server.stop()

    failures = [c for c in results if not c.ok and not c.skipped]
    total_seconds = sum(c.seconds for c in results)

    if args.json:
        print(json.dumps(
            {"checks": [asdict(c) for c in results], "failures": len(failures)}, indent=2
        ))
        return 1 if failures else 0

    for c in results:
        mark = "skip" if c.skipped else ("pass" if c.ok else "FAIL")
        timing = f"{DIM} ({c.seconds:.1f}s){RESET}" if c.seconds >= 0.05 else ""
        print(f"  {_colour(c.ok, c.skipped)}{mark:>4}{RESET}  {c.name:<32}{DIM}{c.detail}{RESET}{timing}")

    print()
    if failures:
        print(f"{RED}{BOLD}{len(failures)} check(s) failed{RESET} {DIM}({total_seconds:.1f}s){RESET}")
        for c in failures:
            print(f"  {RED}->{RESET} {c.name}: {c.detail}")
        return 1
    skipped = sum(1 for c in results if c.skipped)
    tail = f", {skipped} skipped" if skipped else ""
    print(f"{GREEN}{BOLD}all checks passed{RESET} {DIM}({total_seconds:.1f}s{tail}){RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
