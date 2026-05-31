"""Lightweight Ollama client — no langchain_ollama, no heavy ML imports."""
from functools import lru_cache

import httpx
from rich.console import Console

from mariana.utils.config import load_config

MAX_LLM_INPUT_CHARS = 2000
console = Console()


def _msg_role(msg) -> str:
    """Convert a LangChain message object to an Ollama role string."""
    t = getattr(msg, "type", "human")
    if t == "system":
        return "system"
    if t in ("human", "user"):
        return "user"
    return "assistant"


class _Response:
    """Minimal stand-in for AIMessage — has .content attribute."""
    __slots__ = ("content",)

    def __init__(self, content: str):
        self.content = content


class _Chain:
    """Result of ``prompt | OllamaLLM()``. Supports .invoke(vars)."""
    __slots__ = ("_prompt", "_llm")

    def __init__(self, prompt, llm: "OllamaLLM"):
        self._prompt = prompt
        self._llm = llm

    def invoke(self, variables: dict) -> _Response:
        messages = self._prompt.format_messages(**variables)
        ollama_msgs = [{"role": _msg_role(m), "content": m.content} for m in messages]
        return self._llm._call(ollama_msgs)


class OllamaLLM:
    """
    Thin httpx wrapper around Ollama's /api/chat endpoint.

    Supports the LCEL ``|`` operator so existing code like
        (EXTRACT_PROMPT | llm).invoke(vars)
    continues to work unchanged.
    """

    def __init__(self, model: str, base_url: str, temperature: float, num_predict: int):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.num_predict = num_predict

    # Support: prompt | llm  (langchain_core wraps callables in RunnableLambda)
    def __call__(self, input) -> _Response:
        return self.invoke(input)

    # Support: prompt | llm
    def __ror__(self, prompt) -> _Chain:
        return _Chain(prompt, self)

    def _call(self, messages: list[dict]) -> _Response:
        resp = httpx.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.num_predict,
                },
            },
            timeout=120.0,
        )
        resp.raise_for_status()
        content = resp.json().get("message", {}).get("content", "")
        return _Response(content)

    def invoke(self, input) -> _Response:
        """Direct invocation — accepts a string, message list, or PromptValue."""
        if isinstance(input, str):
            messages = [{"role": "user", "content": input}]
        elif isinstance(input, list):
            messages = [{"role": _msg_role(m), "content": m.content} for m in input]
        else:
            # ChatPromptValue or similar — try .to_messages()
            try:
                msgs = input.to_messages()
                messages = [{"role": _msg_role(m), "content": m.content} for m in msgs]
            except AttributeError:
                messages = [{"role": "user", "content": str(input)}]
        return self._call(messages)


@lru_cache(maxsize=1)
def get_planner_llm() -> OllamaLLM:
    cfg = load_config()
    return OllamaLLM(
        model=cfg.planner_model,
        base_url=cfg.ollama_base_url,
        temperature=0.4,
        num_predict=1024,
    )


@lru_cache(maxsize=1)
def get_summarizer_llm() -> OllamaLLM:
    cfg = load_config()
    return OllamaLLM(
        model=cfg.summarizer_model,
        base_url=cfg.ollama_base_url,
        temperature=0.1,
        num_predict=2048,
    )


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
