from __future__ import annotations

import argparse
import threading
import time
import webbrowser

import uvicorn

from .settings import AppSettings
from .webapp import create_app


def _open_browser_later(url: str, delay_s: float = 0.8) -> None:
    def _runner() -> None:
        time.sleep(max(0.0, float(delay_s)))
        try:
            webbrowser.open(url)
        except Exception:
            pass

    threading.Thread(target=_runner, daemon=True).start()


def main() -> None:
    parser = argparse.ArgumentParser(description="Spustí webový frontend Chatbotu Kája přes lokální HTTP server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    settings = AppSettings.load()
    app = create_app(settings)
    config = uvicorn.Config(app=app, host=args.host, port=args.port, log_level="info")
    server = uvicorn.Server(config)
    app.state.server = server

    url = f"http://{args.host}:{args.port}/"
    if not args.no_browser:
        _open_browser_later(url)

    print(f"Chatbot Kája WebRTC běží na {url}")
    server.run()


if __name__ == "__main__":
    main()
