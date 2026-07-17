from __future__ import annotations

import json
import mimetypes
import secrets
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.metadata import PackageNotFoundError, version
from importlib.resources import files
from pathlib import Path, PurePosixPath
from urllib.parse import parse_qs, urlsplit

import cv2
from loguru import logger

from datenwissenschaften.runtime import get_runtime
from datenwissenschaften.settings import UISettings
from datenwissenschaften.ui.control import control_metadata, request_model_reset
from datenwissenschaften.ui.telemetry import get_store


def _datenwissenschaften_version() -> str:
    try:
        return version("datenwissenschaften")
    except PackageNotFoundError:
        return "DEVELOPMENT"


DATENWISSENSCHAFTEN_VERSION = _datenwissenschaften_version()
GENERATED_SOURCE_FILES = (
    ".containerignore",
    "Containerfile",
    "actions.py",
    "app.py",
    "config.yaml",
    "pyproject.toml",
    "ram.py",
    "runner.py",
    "states.py",
    "wrapper.py",
)


def generated_sources(root: Path | None = None) -> list[dict[str, str | int]]:
    source_root = (root or Path.cwd()).resolve()
    result = []
    for relative_path in GENERATED_SOURCE_FILES:
        source_path = source_root / relative_path
        if source_path.is_file():
            result.append(
                {
                    "path": relative_path,
                    "language": _source_language(relative_path),
                    "size": source_path.stat().st_size,
                }
            )
    return result


def generated_source(relative_path: str, root: Path | None = None) -> dict[str, str | int]:
    if relative_path not in GENERATED_SOURCE_FILES:
        raise FileNotFoundError(relative_path)
    source_path = (root or Path.cwd()).resolve() / relative_path
    content = source_path.read_text(encoding="utf-8")
    if relative_path == "config.yaml":
        content = _redact_config_secrets(content)
    return {
        "path": relative_path,
        "language": _source_language(relative_path),
        "size": source_path.stat().st_size,
        "content": content,
    }


def _source_language(path: str) -> str:
    if path.endswith(".py"):
        return "python"
    if path.endswith((".yaml", ".yml")):
        return "yaml"
    if path.endswith(".toml"):
        return "toml"
    if path == "Containerfile":
        return "dockerfile"
    return "text"


def _redact_config_secrets(content: str) -> str:
    lines = []
    in_upload = False
    for line in content.splitlines(keepends=True):
        if line and not line[0].isspace():
            in_upload = line.rstrip() == "upload:"
        if in_upload and line.lstrip().startswith("api_key:"):
            indentation = line[: len(line) - len(line.lstrip())]
            newline = "\n" if line.endswith("\n") else ""
            line = f'{indentation}api_key: "[REDACTED]"{newline}'
        lines.append(line)
    return "".join(lines)


def learned_enemies() -> list[dict[str, str | int]]:
    root = get_runtime().cache_dir / "learned_enemies"
    result = []
    for path in sorted(root.glob("*/*.png")):
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None or image.ndim != 3 or image.shape[2] != 4:
            path.unlink(missing_ok=True)
            continue
        alpha = image[..., 3]
        foreground_pixels = cv2.countNonZero(alpha)
        if foreground_pixels < 16 or foreground_pixels == alpha.size:
            path.unlink(missing_ok=True)
            continue
        game, filename = path.relative_to(root).parts
        result.append(
            {
                "id": path.stem,
                "path": "/".join((game, filename)),
                "game": game,
                "savestate": "",
                "state": "Explorer",
                "size": path.stat().st_size,
            }
        )
    return result


def learned_enemy_path(relative_path: str) -> Path:
    root = (get_runtime().cache_dir / "learned_enemies").resolve()
    candidate = (root / relative_path).resolve()
    if candidate.suffix.lower() != ".png" or not candidate.is_relative_to(root) or not candidate.is_file():
        raise FileNotFoundError(relative_path)
    return candidate


class DashboardServer:
    def __init__(self, settings: UISettings) -> None:
        self.settings = settings
        self._httpd = ThreadingHTTPServer((settings.host, settings.port), _DashboardHandler)
        self._httpd.csrf_token = secrets.token_urlsafe(32)
        self._httpd.ui_settings = settings
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
    browser_host = "127.0.0.1" if settings.host == "0.0.0.0" else settings.host
    logger.info(
        f"Training UI listening on {settings.host}:{settings.port}; "
        f"open http://{browser_host}:{settings.port} locally"
    )
    return _server


class _DashboardHandler(BaseHTTPRequestHandler):
    server_version = "DatenwissenschaftenUI/1.0"

    def do_GET(self) -> None:  # noqa: N802
        request = urlsplit(self.path)
        path = request.path
        if path == "/api/snapshot":
            snapshot = get_store().snapshot()
            snapshot["control"] = {
                **control_metadata(),
                "csrf_token": self.server.csrf_token,
            }
            settings = self.server.ui_settings
            snapshot["server"] = {
                "host": settings.host,
                "port": settings.port,
                "bind_address": f"{settings.host}:{settings.port}",
                "version": DATENWISSENSCHAFTEN_VERSION,
            }
            self._send_json(snapshot)
            return
        if path == "/api/health":
            self._send_json({"status": "ok"})
            return
        if path == "/api/sources":
            self._send_json({"files": generated_sources()})
            return
        if path == "/api/source":
            requested = parse_qs(request.query).get("path", [""])[0]
            try:
                self._send_json(generated_source(requested))
            except (FileNotFoundError, OSError, UnicodeError):
                self.send_error(HTTPStatus.NOT_FOUND, "Generated source file not found")
            return
        if path == "/api/enemies":
            self._send_json({"enemies": learned_enemies()})
            return
        if path == "/api/enemy":
            requested = parse_qs(request.query).get("path", [""])[0]
            try:
                self._send_binary(learned_enemy_path(requested), "image/png")
            except (FileNotFoundError, OSError):
                self.send_error(HTTPStatus.NOT_FOUND, "Learned enemy image not found")
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
        pass

    def _send_json(self, payload: dict, *, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_binary(self, path: Path, content_type: str) -> None:
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-cache")
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
