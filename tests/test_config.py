import os
from pathlib import Path
from types import SimpleNamespace

from mariana.utils.config import MarianaConfig, ENV_FILE, load_config, save_setting


def _patch_config_paths(monkeypatch, tmp_path):
    monkeypatch.setattr("mariana.utils.config.ENV_FILE", tmp_path / ".env")
    monkeypatch.setattr("mariana.utils.config.MarianaConfig.model_config", {
        "env_prefix": "MARIANA_",
        "env_file": str(tmp_path / ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    })


def test_load_config_defaults_when_no_env_file(tmp_path, monkeypatch):
    _patch_config_paths(monkeypatch, tmp_path)
    monkeypatch.delenv("MARIANA_MAX_RESULTS", raising=False)
    monkeypatch.delenv("MARIANA_OUTPUT_DIR", raising=False)
    load_config.cache_clear()

    cfg = load_config()

    assert cfg.searxng_port == 8080
    assert cfg.planner_model == "gemma3:1b"
    assert cfg.summarizer_model == "gemma3:1b"


def test_save_setting_and_load_config_roundtrip(tmp_path, monkeypatch):
    _patch_config_paths(monkeypatch, tmp_path)
    load_config.cache_clear()

    save_setting("MARIANA_MAX_RESULTS", "7")
    load_config.cache_clear()
    cfg = load_config()

    assert cfg.max_results == 7
    assert (tmp_path / ".env").read_text().strip() == "MARIANA_MAX_RESULTS=7"


def test_output_dir_expands_from_env(monkeypatch):
    monkeypatch.setenv("MARIANA_OUTPUT_DIR", "~/mariana_test_output")
    load_config.cache_clear()
    cfg = load_config()
    assert str(cfg.output_dir).startswith(str(Path.home()))
    assert "~" not in str(cfg.output_dir)


def test_search_delay_defaults_to_two_seconds(monkeypatch):
    _patch_config_paths(monkeypatch, tmp_path:=Path("/tmp"))
    monkeypatch.delenv("MARIANA_SEARCH_DELAY_SECONDS", raising=False)
    load_config.cache_clear()
    cfg = load_config()
    assert cfg.search_delay_seconds == 2.0
