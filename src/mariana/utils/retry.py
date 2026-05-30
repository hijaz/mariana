"""LLM retry decorator with exponential back-off via tenacity."""
import functools
from typing import Callable, TypeVar

from rich.console import Console
from tenacity import retry, stop_after_attempt, wait_exponential

console = Console()

F = TypeVar("F", bound=Callable)


def with_llm_retry(fn: F) -> F:
    """Wrap a function so it retries up to 3 times on any exception."""

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    return wrapper  # type: ignore[return-value]
