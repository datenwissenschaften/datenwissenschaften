from __future__ import annotations

import json
import mimetypes
import secrets
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path, PurePosixPath

from loguru import logger

from datenwissenschaften.settings import UISettings
from datenwissenschaften.ui.control import control_metadata, request_model_reset
from datenwissenschaften.ui.telemetry import get_store


class DashboardServer:
    def __init__(self, settings: UISettings) -> None:
        self.settings = settings
        self._httpd = ThreadingHTTPServer((settings.host, settings.port), _DashboardHandler)
        self._httpd.csrf_token = secrets.token_urlsafe(32)
        self._thread = threading.Thread(target=self._httpd.serve_forever, name="training-ui", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()


_server: DashboardServer | None = None
_server_lock = threading.Lock()


def start_ui(settings: UISettings) -> DashboardServer | None:
    global _server
    if not settings.enabled:
        return None
    with _server_lock:
        get_store().resize(settings.max_episodes)
        if _server is not None:
            return _server
        try:
            _server = DashboardServer(settings)
            _server.start()
        except OSError as error:
            logger.error(f"Could not start training UI on {settings.host}:{settings.port}: {error}")
            return None
    logger.info(f"Training UI available at http://{settings.host}:{settings.port}")
    return _server


class _DashboardHandler(BaseHTTPRequestHandler):
    server_version = "DatenwissenschaftenUI/1.0"

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.partition("?")[0]
        if path == "/api/snapshot":
            snapshot = get_store().snapshot()
            snapshot["control"] = {
                **control_metadata(),
                "csrf_token": self.server.csrf_token,
            }
            self._send_json(snapshot)
            return
        if path == "/api/health":
            self._send_json({"status": "ok"})
            return
        self._send_asset(path)

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.partition("?")[0]
        if path != "/api/model/reset":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if self.headers.get("X-CSRF-Token") != self.server.csrf_token:
            self.send_error(HTTPStatus.FORBIDDEN, "Invalid CSRF token")
            return
        if self.headers.get_content_type() != "application/json":
            self.send_error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "Expected application/json")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > 4_096:
                raise ValueError("Invalid request size")
            payload = json.loads(self.rfile.read(length))
            game = payload.get("game") if isinstance(payload, dict) else None
            if not isinstance(game, str) or not game:
                raise ValueError("Missing game")
            request_model_reset(game)
        except (json.JSONDecodeError, ValueError) as error:
            self._send_json({"error": str(error)}, status=HTTPStatus.BAD_REQUEST)
            return
        except RuntimeError as error:
            self._send_json({"error": str(error)}, status=HTTPStatus.CONFLICT)
            return
        logger.warning(f"Model reset requested from training UI for {game}")
        self._send_json({"status": "reset_pending", "game": game}, status=HTTPStatus.ACCEPTED)

    def log_message(self, format: str, *args) -> None:
        logger.debug(f"Training UI: {format % args}")

    def _send_json(self, payload: dict, *, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_asset(self, requested_path: str) -> None:
        relative = requested_path.lstrip("/") or "index.html"
        safe_path = PurePosixPath(relative)
        if ".." in safe_path.parts:
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        root = files("datenwissenschaften.ui").joinpath("static")
        asset = root.joinpath(*safe_path.parts)
        if not asset.is_file():
            asset = root.joinpath("index.html")
        try:
            body = asset.read_bytes()
        except (FileNotFoundError, OSError):
            self.send_error(HTTPStatus.NOT_FOUND, "Dashboard assets are not installed")
            return

        content_type = mimetypes.guess_type(Path(asset.name).name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Cache-Control", "no-cache" if asset.name == "index.html" else "public, max-age=31536000")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
