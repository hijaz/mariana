import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Runtime state directory — PID files, logs, output reports.
# Also stores the local .env file for user settings.
RUNTIME_DIR = Path.home() / ".mariana"
ENV_FILE = RUNTIME_DIR / ".env"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ENV_FILE = PROJECT_ROOT / ".env"


class MarianaConfig(BaseSettings):
    """All settings are read from environment variables or a .env file.
    Prefix every variable with MARIANA_ (e.g. MARIANA_PLANNER_MODEL=gemma3:1b).
    """

    model_config = SettingsConfigDict(
        env_prefix="MARIANA_",
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    searxng_port: int = 8080
    ollama_base_url: str = "http://localhost:11434"
    planner_model: str = "gemma3:1b"
    summarizer_model: str = "gemma3:1b"
    max_results: int = 5
    num_subquestions: int = 3
    max_iterations: int = 3
    min_sections: int = 3
    max_sections: int = 4
    max_page_chars: int = 20000
    search_delay_seconds: float = 1.0
    max_section_words: int = 600
    output_dir: Path = RUNTIME_DIR / "output"

    @field_validator("output_dir", mode="before")
    @classmethod
    def expand_path(cls, v: object) -> Path:
        return Path(str(v)).expanduser()


def _write_env_var(key: str, value: str) -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    lines = []
    if ENV_FILE.exists():
        lines = ENV_FILE.read_text(encoding="utf-8").splitlines()

    key_eq = f"{key}="
    found = False
    for index, line in enumerate(lines):
        if line.startswith(key_eq):
            lines[index] = f"{key}={value}"
            found = True
            break
    if not found:
        lines.append(f"{key}={value}")
    ENV_FILE.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _load_dotenv_files() -> None:
    """Load project .env then runtime ~/.mariana/.env into the process environment."""
    if PROJECT_ENV_FILE.exists():
        load_dotenv(dotenv_path=PROJECT_ENV_FILE, override=False)
    if ENV_FILE.exists():
        load_dotenv(dotenv_path=ENV_FILE, override=True)


@lru_cache(maxsize=1)
def load_config() -> MarianaConfig:
    """Return the config singleton. Reads project .env, ~/.mariana/.env, and env vars once."""
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    _load_dotenv_files()
    return MarianaConfig()


def save_setting(name: str, value: str) -> None:
    """Persist a single environment-style setting to ~/.mariana/.env."""
    _write_env_var(name, value)
    load_config.cache_clear()


def ensure_output_dir(config: MarianaConfig) -> Path:
    """Create output_dir if it doesn't exist. Return the Path."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    return config.output_dir
