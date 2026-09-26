"""FastAPI application: REST, WebSocket feed, and static frontend hosting."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..bus import EventBus
from ..config import Config
from ..db import Database, summarise
from ..ingest.server import IngestServer
from ..pipeline import Pipeline, _public_turn
from ..schemas import (
    HealthResponse,
    ManualTurnRequest,
    StatsResponse,
    Turn,
)

log = logging.getLogger("friday.api")

WEB_DIST = Path(__file__).resolve().parent.parent.parent / "web" / "dist"


def create_app(
    cfg: Config | None = None,
    *,
    db: Database | None = None,
    pipeline: Pipeline | None = None,
    bus: EventBus | None = None,
    start_ingest: bool = True,
) -> FastAPI:
    cfg = cfg or Config.load()
    cfg.ensure_dirs()

    db = db or Database(cfg.db_path)
    bus = bus or EventBus()
    pipeline = pipeline or Pipeline(cfg, db, bus)
    ingest = IngestServer(cfg, pipeline)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        bus.bind_loop(asyncio.get_running_loop())
        # A turn still mid-pipeline at startup belongs to a process that died.
        # Fail it now rather than showing a spinner that never resolves.
        for turn_id in db.recover_interrupted():
            log.warning("recovered interrupted turn %s", turn_id)
        pipeline.start()
        if start_ingest:
            try:
                await ingest.start()
            except OSError as exc:
                log.error("could not start ingest listener: %s", exc)
        try:
            yield
        finally:
            if start_ingest:
                await ingest.stop()
            pipeline.stop()

    app = FastAPI(title="FRIDAY", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Exposed for tests and for the __main__ runner.
    app.state.cfg = cfg
    app.state.db = db
    app.state.bus = bus
    app.state.pipeline = pipeline
    app.state.ingest = ingest

    # ------------------------------------------------------------- turns API
    @app.get("/api/turns", response_model=list[Turn])
    async def list_turns(limit: int = 50, offset: int = 0) -> list[dict]:
        limit = max(1, min(limit, 500))
        return [_public_turn(t) for t in db.list_turns(limit=limit, offset=offset)]

    @app.get("/api/turns/{turn_id}", response_model=Turn)
    async def get_turn(turn_id: str) -> dict:
        turn = db.get_turn(turn_id)
        if turn is None:
            raise HTTPException(404, "no such turn")
        return _public_turn(turn)

    @app.post("/api/turns", response_model=Turn, status_code=201)
    async def create_manual_turn(body: ManualTurnRequest) -> dict:
        """Text-only turn. Lets the UI be driven with no audio at all."""

        turn = db.create_turn(source="manual")
        pipeline.bus.publish_threadsafe(
            {"event": "turn.created", "turn": _public_turn(turn)}
        )
        pipeline.submit_text(turn["id"], body.text)
        return _public_turn(turn)

    @app.delete("/api/turns/{turn_id}", status_code=204)
    async def delete_turn(turn_id: str) -> None:
        turn = db.get_turn(turn_id)
        if turn is None:
            raise HTTPException(404, "no such turn")
        if turn.get("audio_path"):
            with contextlib.suppress(OSError):
                Path(turn["audio_path"]).unlink(missing_ok=True)
        db.delete_turn(turn_id)

    @app.get("/api/audio/{turn_id}")
    async def get_audio(turn_id: str) -> FileResponse:
        turn = db.get_turn(turn_id)
        if not turn or not turn.get("audio_path"):
            raise HTTPException(404, "no audio for this turn")
        path = Path(turn["audio_path"])
        if not path.exists():
            raise HTTPException(404, "audio file is gone")
        return FileResponse(
            path,
            media_type="audio/wav",
            filename=path.name,
            headers={"Cache-Control": "public, max-age=86400"},
        )

    # ------------------------------------------------------------ meta API
    @app.get("/api/stats", response_model=StatsResponse)
    async def stats() -> dict:
        return summarise(db.list_turns(limit=200))

    @app.get("/api/health", response_model=HealthResponse)
    async def health() -> dict:
        istats = ingest.stats()
        stt_state = "ready" if pipeline.stt.loaded else "lazy"
        llm_state = "ready" if pipeline.agent.loaded else "lazy"
        degraded = not istats["listening"]
        return {
            "status": "degraded" if degraded else "ok",
            "ingest": "listening" if istats["listening"] else "down",
            "stt": stt_state,
            "llm": llm_state,
            "device_connected": pipeline.device_connected,
            "turns": db.count(),
        }

    @app.get("/api/ingest/stats")
    async def ingest_stats() -> dict:
        return {
            **ingest.stats(),
            "queue_depth": pipeline.queue_depth,
            "stt_device": pipeline.stt.device,
            "stt_compute_type": pipeline.stt.compute_type,
            "llm_loaded": pipeline.agent.loaded,
        }

    @app.get("/api/system")
    async def system_status() -> dict:
        """Same payload the system_status tool returns, for the vitals strip."""
        from ..tools import _system_status

        return _system_status(cfg)

    # ----------------------------------------------------------- websocket
    @app.websocket("/ws")
    async def ws_feed(websocket: WebSocket) -> None:
        await websocket.accept()
        queue = await bus.subscribe()
        log.info("websocket client connected (%d total)", bus.subscriber_count)

        async def pump() -> None:
            # Replay recent history so a late joiner is not looking at a blank page.
            for turn in reversed(db.list_turns(limit=30)):
                await websocket.send_json(
                    {"event": "turn.created", "turn": _public_turn(turn)}
                )
            while True:
                await websocket.send_json(await queue.get())

        async def listen() -> None:
            # The client never sends us anything, so this exists purely to notice
            # the socket going away. A handler that only ever sends cannot learn
            # the connection is gone: the subscriber leaks when a browser tab is
            # closed, and uvicorn blocks its graceful shutdown on this task until
            # the operator has to signal the process a second time.
            while True:
                await websocket.receive()

        pumping = asyncio.create_task(pump())
        listening = asyncio.create_task(listen())
        try:
            await asyncio.wait(
                (pumping, listening), return_when=asyncio.FIRST_COMPLETED
            )
        except (WebSocketDisconnect, RuntimeError):
            pass
        except Exception:  # noqa: BLE001
            log.debug("websocket feed ended", exc_info=True)
        finally:
            for task in (pumping, listening):
                task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await asyncio.gather(pumping, listening, return_exceptions=True)
            await bus.unsubscribe(queue)
            with contextlib.suppress(RuntimeError):
                await websocket.close()
            log.info("websocket client disconnected")

    # -------------------------------------------------------------- frontend
    if (WEB_DIST / "assets").exists():
        app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

        @app.get("/{full_path:path}")
        async def spa(full_path: str) -> Any:
            if full_path.startswith("api/"):
                return JSONResponse({"error": "not found"}, status_code=404)
            candidate = WEB_DIST / full_path
            if full_path and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(WEB_DIST / "index.html")
    else:

        @app.get("/")
        async def no_frontend() -> Any:
            return JSONResponse(
                {
                    "message": "FRIDAY backend is running. The frontend has not been built.",
                    "hint": "cd web && npm install && npm run build",
                    "api": ["/api/turns", "/api/health", "/api/ingest/stats", "/ws"],
                }
            )

    return app
