import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

from mariana.utils.config import RUNTIME_DIR, load_config

MINIMAL_SETTINGS = """use_default_settings: true
general:
  instance_name: "Mariana Search"
search:
  safe_search: 0
  formats:
    - html
    - json
server:
  secret_key: "mariana-local-only-key"
  bind_address: "0.0.0.0"
  port: {PORT}
engines:
  - name: google
    engine: google
    shortcut: g
    disabled: false
  - name: duckduckgo
    engine: duckduckgo
    shortcut: d
    disabled: false
  - name: bing
    engine: bing
    shortcut: b
    disabled: false
  - name: wikipedia
    engine: wikipedia
    shortcut: wp
    disabled: false
  - name: arxiv
    engine: arxiv
    shortcut: ar
    disabled: false
  - name: brave
    engine: brave
    shortcut: br
    disabled: false
"""


def _kill_pid(pid: int) -> None:
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)
        else:
            os.kill(pid, signal.SIGTERM)
    except Exception:
        return


class SearXNGManager:
    def __init__(self) -> None:
        self.config = load_config()
        self.settings_path = RUNTIME_DIR / "searxng_settings.yml"
        self.pid_file = RUNTIME_DIR / "searxng.pid"
        self.log_file = RUNTIME_DIR / "searxng.log"
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

    def ensure_running(self) -> int:
        port = self.config.searxng_port
        if self._is_responsive(port):
            return port

        port = self._find_free_port(start=port)
        self._write_settings(port)
        self._start_process(port)
        self._wait_until_ready(port)
        self.config.searxng_port = port
        return port

    def stop(self) -> None:
        if not self.pid_file.exists():
            return

        try:
            pid = int(self.pid_file.read_text().strip())
            _kill_pid(pid)
        except Exception:
            pass

        try:
            self.pid_file.unlink()
        except FileNotFoundError:
            pass

    def _find_free_port(self, start: int = 8080) -> int:
        for port in range(start, start + 20):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    sock.bind(("127.0.0.1", port))
                except OSError:
                    continue
                return port
        raise RuntimeError("No free port found in range 8080-8099")

    def _is_responsive(self, port: int) -> bool:
        try:
            resp = httpx.get(f"http://localhost:{port}", timeout=2.0)
            return resp.status_code >= 0
        except Exception:
            return False

    def _write_settings(self, port: int) -> None:
        if self.settings_path.exists():
            return
        self.settings_path.write_text(MINIMAL_SETTINGS.replace("{PORT}", str(port)))

    def _start_process(self, port: int) -> None:
        with open(self.log_file, "w", encoding="utf-8") as log:
            env = {**os.environ, "SEARXNG_SETTINGS_PATH": str(self.settings_path), "SEARXNG_PORT": str(port)}
            proc = subprocess.Popen(
                [sys.executable, "-m", "searx.webapp"],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
            Path(self.pid_file).write_text(str(proc.pid))

    def _wait_until_ready(self, port: int, timeout: int = 20) -> None:
        start = time.time()
        while time.time() - start < timeout:
            if self._is_responsive(port):
                return
            time.sleep(0.5)
        raise RuntimeError(
            f"SearXNG did not become responsive on port {port} within {timeout} seconds."
            f" See {self.log_file} for details."
        )
