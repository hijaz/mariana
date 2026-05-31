"""
In-process event bus for streaming research progress to the UI.

Usage from anywhere in the pipeline:
    from mariana.utils.events import emit
    emit("url_visited", {"url": "https://...", "doc_id": "..."})

The FastAPI SSE endpoint subscribes with:
    async for event in subscribe(doc_id):
        yield event
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from collections import deque
from typing import Any, AsyncIterator

# ── Global state ───────────────────────────────────────────────────────────────

# All active SSE subscribers: {doc_id: [asyncio.Queue, ...]}
_subscribers: dict[str, list[asyncio.Queue]] = {}
_subscribers_lock = threading.Lock()

# Replay buffer per doc_id so late-joining clients get recent history
_replay: dict[str, deque] = {}
_REPLAY_SIZE = 200


# ── Emit (called from sync pipeline threads) ──────────────────────────────────

def emit(event_type: str, data: dict[str, Any], doc_id: str | None = None) -> None:
    """Emit an event. Safe to call from any thread, including the research pipeline."""
    payload = {"type": event_type, "ts": time.time(), **data}
    if doc_id is not None:
        payload["doc_id"] = doc_id

    key = doc_id or "__global__"

    # Store in replay buffer
    with _subscribers_lock:
        if key not in _replay:
            _replay[key] = deque(maxlen=_REPLAY_SIZE)
        _replay[key].append(payload)
        queues = list(_subscribers.get(key, []))
        # Also broadcast to global subscribers
        global_queues = list(_subscribers.get("__global__", [])) if key != "__global__" else []

    encoded = json.dumps(payload, default=str)
    for q in queues + global_queues:
        # thread-safe put into asyncio queue
        try:
            loop = q._loop  # type: ignore[attr-defined]
            loop.call_soon_threadsafe(q.put_nowait, encoded)
        except Exception:
            pass


# ── Subscribe (called from async FastAPI handlers) ────────────────────────────

async def subscribe(doc_id: str | None = None) -> AsyncIterator[str]:
    """Async generator that yields SSE-encoded JSON events for a doc (or all docs)."""
    key = doc_id or "__global__"
    q: asyncio.Queue = asyncio.Queue(maxsize=500)
    q._loop = asyncio.get_event_loop()  # type: ignore[attr-defined]

    # Send replay buffer first
    with _subscribers_lock:
        if key not in _subscribers:
            _subscribers[key] = []
        _subscribers[key].append(q)
        replay_events = list(_replay.get(key, []))
        if key != "__global__":
            replay_events += list(_replay.get("__global__", []))
        replay_events.sort(key=lambda e: e.get("ts", 0))

    try:
        for event in replay_events:
            yield json.dumps(event, default=str)

        while True:
            try:
                msg = await asyncio.wait_for(q.get(), timeout=15.0)
                yield msg
            except asyncio.TimeoutError:
                # Send keepalive comment so the connection stays open
                yield ": keepalive\n\n"
    finally:
        with _subscribers_lock:
            try:
                _subscribers[key].remove(q)
            except ValueError:
                pass


def get_replay(doc_id: str | None = None) -> list[dict]:
    """Return the buffered events for a doc (for HTTP polling fallback)."""
    key = doc_id or "__global__"
    with _subscribers_lock:
        return list(_replay.get(key, []))
