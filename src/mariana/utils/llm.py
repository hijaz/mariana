from functools import lru_cache
from typing import Any

import httpx
from langchain_ollama import ChatOllama
from rich.console import Console

from mariana.utils.config import load_config

MAX_LLM_INPUT_CHARS = 2000
console = Console()


@lru_cache(maxsize=1)
def get_planner_llm() -> ChatOllama:
    cfg = load_config()
    return ChatOllama(
        model=cfg.planner_model,
        base_url=cfg.ollama_base_url,
        temperature=0.4,
        num_predict=1024,
    )


@lru_cache(maxsize=1)
def get_summarizer_llm() -> ChatOllama:
    cfg = load_config()
    return ChatOllama(
        model=cfg.summarizer_model,
        base_url=cfg.ollama_base_url,
        temperature=0.1,
        num_predict=2048,
    )


def truncate_for_llm(text: str, label: str = "input") -> str:
    """Truncate text to MAX_LLM_INPUT_CHARS and warn if truncated."""
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
            if not models:
                return False, "no models available on Ollama"
            return True, models
        if isinstance(data, dict):
            if "models" in data and isinstance(data["models"], list):
                models = [str(item.get("model", item.get("name", ""))) for item in data["models"] if isinstance(item, dict)]
                if not models:
                    return False, "no models available on Ollama"
                return True, models
            if "tags" in data and isinstance(data["tags"], list):
                models = [str(item) for item in data["tags"]]
                if not models:
                    return False, "no models available on Ollama"
                return True, models
        return False, "unexpected Ollama response"
    except Exception as exc:
        return False, str(exc)
