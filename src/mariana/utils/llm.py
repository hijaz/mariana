"""LLM client — langchain_ollama.ChatOllama with LangSmith tracing support."""
from functools import lru_cache
from pathlib import Path

import httpx
from langchain_ollama import ChatOllama
from rich.console import Console

from mariana.utils.config import load_config

MAX_LLM_INPUT_CHARS = 2000
console = Console()


def _log_mem_pressure() -> None:
    """Print available system memory so OOM pressure is visible in logs."""
    try:
        meminfo = Path("/proc/meminfo").read_text()
        avail = next(
            int(line.split()[1])
            for line in meminfo.splitlines()
            if line.startswith("MemAvailable:")
        )
        swap_free = next(
            (int(line.split()[1]) for line in meminfo.splitlines() if line.startswith("SwapFree:")),
            0,
        )
        avail_mb = avail // 1024
        swap_mb = swap_free // 1024
        level = "red" if avail_mb < 512 else "yellow" if avail_mb < 1024 else "dim"
        console.print(f"  [{level}]sys mem: {avail_mb} MB RAM free, {swap_mb} MB swap free[/]")
    except Exception:
        pass


@lru_cache(maxsize=1)
def get_llm() -> ChatOllama:
    """Single shared ChatOllama instance. LangSmith traces automatically when
    LANGCHAIN_TRACING_V2=true and LANGCHAIN_API_KEY are set in the environment."""
    cfg = load_config()
    return ChatOllama(
        model=cfg.planner_model,
        base_url=cfg.ollama_base_url,
        temperature=0.3,
        num_predict=512,
        num_ctx=2048,
    )


def get_planner_llm() -> ChatOllama:
    """Alias for get_llm() — kept for call-site compatibility."""
    return get_llm()


def get_summarizer_llm() -> ChatOllama:
    """Alias for get_llm() — kept for call-site compatibility."""
    return get_llm()


def truncate_for_llm(text: str, label: str = "input") -> str:
    if len(text) <= MAX_LLM_INPUT_CHARS:
        return text
    console.print(
        f"  [dim][LLM] truncating {label} from {len(text)} → {MAX_LLM_INPUT_CHARS} chars[/]"
    )
    return text[:MAX_LLM_INPUT_CHARS]


def check_ollama() -> tuple[bool, list[str] | str]:
    cfg = load_config()
    try:
        resp = httpx.get(f"{cfg.ollama_base_url}/api/tags", timeout=5.0)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            models = [str(item) for item in data]
            return (True, models) if models else (False, "no models available on Ollama")
        if isinstance(data, dict):
            if "models" in data and isinstance(data["models"], list):
                models = [
                    str(item.get("model", item.get("name", "")))
                    for item in data["models"]
                    if isinstance(item, dict)
                ]
                return (True, models) if models else (False, "no models available on Ollama")
            if "tags" in data and isinstance(data["tags"], list):
                models = [str(item) for item in data["tags"]]
                return (True, models) if models else (False, "no models available on Ollama")
        return False, "unexpected Ollama response"
    except Exception as exc:
        return False, str(exc)

