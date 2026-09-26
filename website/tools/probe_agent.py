"""Ad-hoc harness: exercise the agent against a list of questions."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.agent import Agent  # noqa: E402
from server.config import Config  # noqa: E402
from server.tools import build_registry  # noqa: E402
from server.tools.weather import WeatherClient  # noqa: E402

QUESTIONS = [
    "What is 7 times 23?",
    "What is the weather in Pune right now?",
    "What is 144 divided by 12?",
    "What time is it right now?",
    "What is 2 to the power of 10?",
    "How much memory does this machine have?",
    "blah blah wubble frobnicate",
]


def main() -> int:
    cfg = Config.load()
    wc = WeatherClient()
    reg = build_registry(cfg, wc)
    ag = Agent(cfg, reg)
    ag.load()

    for q in QUESTIONS:
        t0 = time.monotonic()
        r = ag.answer(q)
        dt = time.monotonic() - t0
        print("=" * 72)
        print("Q:", q)
        print("A:", repr(r.text))
        for c in r.tool_calls:
            detail = c.error or "ok"
            extra = c.result.get("result") if isinstance(c.result, dict) else None
            print(f"   tool: {c.name} {c.arguments} -> {detail} {extra or ''} ({c.duration_ms}ms)")
        print(f"   {dt:.2f}s rounds={r.rounds} err={r.error}")

    ag.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
