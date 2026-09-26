"""In-process publish/subscribe that feeds the WebSocket feed.

The pipeline runs on a worker thread while the HTTP layer lives on the event
loop, so this is a thread-safe fan-out rather than anything asyncio-native.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

log = logging.getLogger("friday.bus")


class EventBus:
    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self._lock = asyncio.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Remember the loop that will service subscriber queues."""
        self._loop = loop

    async def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=256)
        async with self._lock:
            self._subscribers.add(q)
        return q

    async def unsubscribe(self, q: asyncio.Queue[dict[str, Any]]) -> None:
        async with self._lock:
            self._subscribers.discard(q)

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    def publish_threadsafe(self, event: dict[str, Any]) -> None:
        """Fan an event out to every subscriber. Safe to call from any thread."""
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        asyncio.run_coroutine_threadsafe(self._deliver(event), loop)

    async def publish(self, event: dict[str, Any]) -> None:
        await self._deliver(event)

    async def _deliver(self, event: dict[str, Any]) -> None:
        async with self._lock:
            targets = list(self._subscribers)
        for q in targets:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                # A stalled client must never block the pipeline.
                log.warning("dropping event for a slow WebSocket subscriber")


def make_listener(
    bus: EventBus, on_event: Callable[[dict[str, Any]], None]
) -> Callable[[], None]:
    """Create a subscribe callback for use as a FastAPI dependency."""

    async def listener() -> asyncio.Queue[dict[str, Any]]:
        q = await bus.subscribe()
        try:
            while True:
                event = await q.get()
                on_event(event)
        finally:
            await bus.unsubscribe(q)

    return listener
