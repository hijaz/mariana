from pathlib import Path
from types import SimpleNamespace

from mariana.utils.searxng import SearXNGManager


def test_find_free_port_returns_int():
    manager = SearXNGManager()
    port = manager._find_free_port(8080)
    assert isinstance(port, int)
    assert 8080 <= port < 8100


def test_is_responsive_returns_false_when_nothing_listening():
    manager = SearXNGManager()
    assert manager._is_responsive(59999) is False


def test_write_settings_creates_file(tmp_path, monkeypatch):
    monkeypatch.setattr("mariana.utils.searxng.RUNTIME_DIR", tmp_path)
    monkeypatch.setattr("mariana.utils.searxng.load_config", lambda: SimpleNamespace(searxng_port=8080))
    manager = SearXNGManager()

    manager._write_settings(8080)
    assert manager.settings_path.exists()
    assert "port: 8080" in manager.settings_path.read_text()


def test_write_settings_does_not_overwrite_existing_file(tmp_path, monkeypatch):
    monkeypatch.setattr("mariana.utils.searxng.RUNTIME_DIR", tmp_path)
    monkeypatch.setattr("mariana.utils.searxng.load_config", lambda: SimpleNamespace(searxng_port=8080))
    manager = SearXNGManager()

    manager.settings_path.write_text("existing\n")
    manager._write_settings(8080)
    assert manager.settings_path.read_text() == "existing\n"


def test_ensure_running_calls_start_process_when_not_responsive(monkeypatch):
    monkeypatch.setattr("mariana.utils.searxng.load_config", lambda: SimpleNamespace(searxng_port=8080))
    manager = SearXNGManager()
    monkeypatch.setattr(manager, "_is_responsive", lambda port: False)
    monkeypatch.setattr(manager, "_find_free_port", lambda start=8080: 8081)
    calls = []
    monkeypatch.setattr(manager, "_write_settings", lambda port: calls.append(("write", port)))
    monkeypatch.setattr(manager, "_start_process", lambda port: calls.append(("start", port)))
    monkeypatch.setattr(manager, "_wait_until_ready", lambda port: calls.append(("wait", port)))

    port = manager.ensure_running()

    assert port == 8081
    assert ("start", 8081) in calls


def test_stop_silently_handles_missing_pid_file(tmp_path, monkeypatch):
    monkeypatch.setattr("mariana.utils.searxng.RUNTIME_DIR", tmp_path)
    manager = SearXNGManager()
    manager.stop()
