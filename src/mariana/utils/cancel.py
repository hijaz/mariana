"""
Module-level cancellation flag shared between the FastAPI server and the
synchronous pipeline threads.  The server sets `stop_event` when
POST /api/research/stop is called; pipeline nodes call `is_cancelled()`.
"""
import threading

# Default: a never-set event so `is_cancelled()` always returns False
# when running outside the UI server (e.g. CLI).
stop_event: threading.Event = threading.Event()


def is_cancelled() -> bool:
    return stop_event.is_set()
