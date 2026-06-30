from __future__ import annotations

import atexit
import json
import math
import threading
import time
from collections import deque
from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger


def _timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


class TelemetryStore:
    def __init__(self, max_episodes: int = 5_000) -> None:
        self._lock = threading.RLock()
        self._write_lock = threading.Lock()
        self._episodes: deque[dict[str, Any]] = deque(maxlen=max_episodes)
        self._generations: deque[dict[str, Any]] = deque(maxlen=1_000)
        self._metadata: dict[str, Any] = {}
        self._started_at = _timestamp()
        self._sequence = 0
        self._history_path: Path | None = None
        self._persist_event = threading.Event()
        self._writer_thread: threading.Thread | None = None

    def resize(self, max_episodes: int) -> None:
        with self._lock:
            self._episodes = deque(self._episodes, maxlen=max_episodes)

    def configure_history(self, path: Path) -> None:
        path = path.expanduser().resolve()
        self.flush()
        with self._lock:
            if self._history_path == path:
                return
            self._history_path = path
            self._episodes.clear()
            self._generations.clear()
            self._metadata.clear()
            self._sequence = 0
            self._started_at = _timestamp()
            self._load_history_locked()
            if self._writer_thread is None:
                self._writer_thread = threading.Thread(
                    target=self._persist_loop,
                    name="training-history",
                    daemon=True,
                )
                self._writer_thread.start()

    def publish_episode(self, values: dict[str, Any]) -> None:
        with self._lock:
            self._sequence += 1
            self._episodes.append(
                {
                    "index": self._sequence,
                    "timestamp": _timestamp(),
                    **_json_value(values),
                }
            )
            self._mark_dirty()

    def publish_generation(self, values: dict[str, Any]) -> None:
        with self._lock:
            self._generations.append({"timestamp": _timestamp(), **_json_value(values)})
            self._mark_dirty()

    def publish_metadata(self, section: str, values: dict[str, Any], *, replace: bool = False) -> None:
        with self._lock:
            serialized = _json_value(values)
            if replace:
                self._metadata[section] = serialized
            else:
                current = self._metadata.setdefault(section, {})
                current.update(serialized)
            self._mark_dirty()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return deepcopy(
                {
                    "status": "live",
                    "started_at": self._started_at,
                    "updated_at": _timestamp(),
                    "episodes": list(self._episodes),
                    "generations": list(self._generations),
                    "metadata": self._metadata,
                }
            )

    def flush(self) -> None:
        with self._write_lock:
            with self._lock:
                if self._history_path is None:
                    return
                path = self._history_path
                payload = self._snapshot_locked()
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary_path = path.with_name(f".{path.name}.tmp")
                temporary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                temporary_path.replace(path)
            except OSError as error:
                logger.warning(f"Could not persist training UI history to {path}: {error}")

    def reset_for_restart(self, delete_model_directory: Callable[[], None]) -> None:
        """Delete model files without allowing the history writer to recreate them mid-reset."""
        with self._write_lock:
            self._persist_event.clear()
            delete_model_directory()
            with self._lock:
                self._episodes.clear()
                self._generations.clear()
                self._sequence = 0
                self._started_at = _timestamp()
                self._mark_dirty()

    def _snapshot_locked(self) -> dict[str, Any]:
        return {
            "version": 1,
            "started_at": self._started_at,
            "updated_at": _timestamp(),
            "episodes": list(self._episodes),
            "generations": list(self._generations),
            "metadata": self._metadata,
        }

    def _load_history_locked(self) -> None:
        if self._history_path is None or not self._history_path.is_file():
            return
        try:
            payload = json.loads(self._history_path.read_text(encoding="utf-8"))
            episodes = payload.get("episodes", [])
            generations = payload.get("generations", [])
            metadata = payload.get("metadata", {})
            if not isinstance(episodes, list) or not isinstance(generations, list) or not isinstance(metadata, dict):
                raise ValueError("history fields have invalid types")
            self._episodes.extend(item for item in episodes if isinstance(item, dict))
            self._generations.extend(item for item in generations if isinstance(item, dict))
            self._metadata.update(metadata)
            self._sequence = max(
                (item.get("index", 0) for item in self._episodes if isinstance(item.get("index"), int)),
                default=0,
            )
            self._started_at = payload.get("started_at") or self._started_at
            logger.info(f"Loaded {len(self._episodes)} training episodes from {self._history_path}")
        except (OSError, ValueError, json.JSONDecodeError) as error:
            logger.warning(f"Ignoring unreadable training UI history {self._history_path}: {error}")

    def _mark_dirty(self) -> None:
        if self._history_path is not None:
            self._persist_event.set()

    def _persist_loop(self) -> None:
        while True:
            self._persist_event.wait()
            time.sleep(0.25)
            self._persist_event.clear()
            self.flush()


_store = TelemetryStore()


def get_store() -> TelemetryStore:
    return _store


def configure_history(path: Path) -> None:
    _store.configure_history(path)


def publish_episode(**values: Any) -> None:
    _store.publish_episode(values)


def publish_generation(**values: Any) -> None:
    _store.publish_generation(values)


def publish_metadata(section: str, values: dict[str, Any], *, replace: bool = False) -> None:
    _store.publish_metadata(section, values, replace=replace)


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _json_value(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(item) for item in value]
    if callable(value):
        return getattr(value, "__name__", str(value))
    return str(value)


atexit.register(_store.flush)
