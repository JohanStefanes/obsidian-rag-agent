"""Native desktop window for the UI.

Runs the FastAPI server in a background thread and shows the same page in a real
macOS window (pywebview wraps the system WebKit view). This is what the
`VaultRAG.app` bundle launches. On macOS `webview.start()` must own the main
thread, so the server runs in a daemon thread beside it.
"""

from __future__ import annotations

import threading
import time
import urllib.request

from .config import Config


def _wait_for_health(url: str, attempts: int = 150) -> bool:
    for _ in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=0.5):
                return True
        except Exception:  # noqa: BLE001 - server not up yet
            time.sleep(0.1)
    return False


def run_desktop(cfg: Config, host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn
    import webview

    from .server import create_app

    config = uvicorn.Config(create_app(cfg), host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    # install_signal_handlers no-ops off the main thread, so this is safe.
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    _wait_for_health(f"http://{host}:{port}/api/status")

    webview.create_window(
        "VaultRAG",
        f"http://{host}:{port}",
        width=980,
        height=760,
        min_size=(680, 560),
    )
    webview.start()  # blocks until the window closes

    server.should_exit = True
    thread.join(timeout=3)
