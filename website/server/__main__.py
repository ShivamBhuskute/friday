"""``python -m server`` entrypoint."""

from __future__ import annotations

import argparse
import logging
import sys

import uvicorn

from .config import Config
from .web.api import create_app


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="friday", description="FRIDAY voice pipeline server")
    ap.add_argument("--host", default=None, help="override the HTTP bind address")
    ap.add_argument("--port", type=int, default=None, help="override the HTTP port")
    ap.add_argument("--ingest-port", type=int, default=None, help="override the TCP ingest port")
    ap.add_argument("--reload", action="store_true", help="auto-reload for development")
    ap.add_argument(
        "--log-level",
        default="info",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
    )
    ap.add_argument(
        "--no-ingest", action="store_true", help="serve HTTP only (useful for tests)"
    )
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)-16s %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = Config.load()
    if args.host:
        cfg.server.host = args.host
    if args.port:
        cfg.server.port = args.port
    if args.ingest_port is not None:
        cfg.ingest.port = args.ingest_port

    app = create_app(cfg, start_ingest=not args.no_ingest)

    print(f"\n  FRIDAY  http://{cfg.server.host}:{cfg.server.port}   ingest tcp/{cfg.ingest.port}\n")
    uvicorn.run(
        app,
        host=cfg.server.host,
        port=cfg.server.port,
        log_level=args.log_level,
        access_log=False,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
