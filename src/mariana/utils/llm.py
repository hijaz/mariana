"""Lightweight Ollama client — no langchain_ollama, no heavy ML imports."""
from functools import lru_cache
from pathlib import Path

import httpx
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


class _SimpleMessage:
    """Minimal message object — .type and .content, no langchain dependency."""
    __slots__ = ("type", "content")

    def __init__(self, role: str, content: str):
        self.type = role
        self.content = content


class SimplePrompt:
    """
    Drop-in replacement for ``ChatPromptTemplate`` with zero langchain deps.

    Usage::

        prompt = SimplePrompt([
            ("system", "You are a {role}."),
            ("human", "Question: {q}"),
        ])
        chain = prompt | llm
        result = chain.invoke({"role": "scientist", "q": "What is fusion?"})
    """

    def __init__(self, messages: list[tuple[str, str]]):
        self._messages = messages

    def format_messages(self, **kwargs) -> list[_SimpleMessage]:
        # Escape braces in VALUES so scraped text with {foo} doesn't crash format_map.
        # Template placeholders like {section_title} still resolve correctly because
        # format_map sees the escaped values as literal text after substitution.
        safe = {k: str(v).replace("{", "{{").replace("}", "}}") for k, v in kwargs.items()}
        result = []
        for role, template in self._messages:
            content = template.format_map(safe)
            result.append(_SimpleMessage(role, content))
        return result

    def __or__(self, other: "OllamaLLM") -> "_Chain":
        return _Chain(self, other)


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

    def __init__(self, model: str, base_url: str, temperature: float, num_predict: int, num_ctx: int | None = None):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.num_predict = num_predict
        self.num_ctx = num_ctx  # None = use Ollama/model default

    # Support: prompt | llm  (langchain_core wraps callables in RunnableLambda)
    def __call__(self, input) -> _Response:
        return self.invoke(input)

    # Support: prompt | llm
    def __ror__(self, prompt) -> _Chain:
        return _Chain(prompt, self)

    def _call(self, messages: list[dict]) -> _Response:
        _log_mem_pressure()
        options: dict = {
            "temperature": self.temperature,
            "num_predict": self.num_predict,
        }
        if self.num_ctx is not None:
            options["num_ctx"] = self.num_ctx
        resp = httpx.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": options,
            },
            timeout=300.0,
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
def get_llm() -> OllamaLLM:
    """Single shared LLM instance — one Ollama session, no session switching."""
    cfg = load_config()
    return OllamaLLM(
        model=cfg.planner_model,
        base_url=cfg.ollama_base_url,
        temperature=0.3,
        num_predict=512,  # all our prompts need ≤200 tokens; 512 gives headroom
        num_ctx=2048,     # explicit: prevents Ollama allocating O(n²) buffers for n=32768
        # Gemma 3 global-attention layers allocate 32768²×heads×2B ≈ 12 GB at the
        # model-default context length; 2048 context costs ~100 MB total.
        # All prompts fit: MAX_LLM_INPUT_CHARS=2000 (~500 tokens) + overhead ≈ 600
        # tokens in, 512 tokens out → 1112 total, well within the 2048 window.
    )


def get_planner_llm() -> OllamaLLM:
    """Alias for get_llm() — kept for call-site compatibility."""
    return get_llm()


def get_summarizer_llm() -> OllamaLLM:
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
