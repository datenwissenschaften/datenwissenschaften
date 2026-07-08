from __future__ import annotations

import atexit
import json
import threading
import time
from collections import deque
from collections.abc import Callable
from copy import deepcopy
from datetime import UTC, datetime
from typing import Any

from loguru import logger

from datenwissenschaften.serialization import to_json_value

try:
    from redis import Redis
    from redis.exceptions import RedisError
except ImportError:
    Redis = None
    RedisError = Exception


def _timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


class TelemetryStore:
    def __init__(self, max_episodes: int | None = None) -> None:
        self._lock = threading.RLock()
        self._write_lock = threading.Lock()
        self._episodes: deque[dict[str, Any]] = deque(maxlen=max_episodes)
        self._generations: deque[dict[str, Any]] = deque(maxlen=1)
        self._metadata: dict[str, Any] = {}
        self._started_at = _timestamp()
        self._sequence = 0
        self._history_key: str | None = None
        self._redis: Any | None = None
        self._history_version = 0
        self._persist_event = threading.Event()
        self._writer_thread: threading.Thread | None = None

    def resize(self, max_episodes: int | None) -> None:
        with self._lock:
            self._episodes = deque(self._episodes, maxlen=max_episodes)

    def configure_history(
        self,
        scope: str,
        *,
        redis_url: str = "redis://127.0.0.1:6379/0",
        key_prefix: str = "datenwissenschaften:history",
    ) -> None:
        history_key = f"{key_prefix.rstrip(':')}:{scope}"
        self.flush()
        with self._lock:
            if self._history_key == history_key:
                return
            if Redis is None:
                raise RuntimeError("Redis history storage requires the 'redis' package. Run `poetry install`.")
            redis_client = Redis.from_url(
                redis_url,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            try:
                redis_client.ping()
            except RedisError as error:
                raise RuntimeError(f"Could not connect to Redis history store at {redis_url}: {error}") from error
            self._redis = redis_client
            self._history_key = history_key
            self._history_version += 1
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
            neat = self._metadata.get("neat", {})
            generation = neat.get("current_generation") if isinstance(neat, dict) else None
            self._episodes.append(
                {
                    "index": self._sequence,
                    "timestamp": _timestamp(),
                    **({"generation": generation} if generation is not None else {}),
                    **to_json_value(values),
                }
            )
            self._mark_dirty()

    def publish_generation(self, values: dict[str, Any]) -> None:
        with self._lock:
            self._generations.append({"timestamp": _timestamp(), **to_json_value(values)})
            self._mark_dirty()

    def publish_metadata(self, section: str, values: dict[str, Any], *, replace: bool = False) -> None:
        with self._lock:
            serialized = to_json_value(values)
            if section == "neat" and "current_generation" in serialized:
                current = self._metadata.get("neat", {})
                previous_generation = current.get("current_generation") if isinstance(current, dict) else None
                if previous_generation is not None and previous_generation != serialized["current_generation"]:
                    self._episodes.clear()
                    self._generations.clear()
            if replace:
                self._metadata[section] = serialized
            else:
                current = self._metadata.setdefault(section, {})
                current.update(serialized)
            self._mark_dirty()

    def clear_metadata(self, section: str, *, clear_history: bool = False) -> None:
        with self._lock:
            removed = self._metadata.pop(section, None)
            if removed is not None and clear_history:
                self._episodes.clear()
                self._generations.clear()
                self._sequence = 0
                self._started_at = _timestamp()
            if removed is not None:
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

    def flush(self, expected_version: int | None = None) -> None:
        with self._write_lock:
            with self._lock:
                if expected_version is not None and expected_version != self._history_version:
                    return
                if self._redis is None or self._history_key is None:
                    return
                redis_client = self._redis
                history_key = self._history_key
                payload = self._snapshot_locked()
            try:
                redis_client.set(history_key, json.dumps(payload, separators=(",", ":")))
            except RedisError as error:
                logger.warning(f"Could not persist training UI history to Redis key {history_key}: {error}")

    def reset_for_restart(self, delete_training_artifacts: Callable[[], None]) -> None:
        with self._write_lock:
            self._persist_event.clear()
            with self._lock:
                redis_client = self._redis
                history_key = self._history_key
            delete_training_artifacts()
            if redis_client is not None and history_key is not None:
                try:
                    redis_client.delete(history_key)
                except RedisError as error:
                    logger.warning(f"Could not delete training UI history from Redis key {history_key}: {error}")
            with self._lock:
                self._episodes.clear()
                self._generations.clear()
                self._sequence = 0
                self._started_at = _timestamp()
                self._history_version += 1

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
        if self._redis is None or self._history_key is None:
            return
        try:
            serialized = self._redis.get(self._history_key)
            if serialized is None:
                return
            payload = json.loads(serialized)
            episodes = payload.get("episodes", [])
            generations = payload.get("generations", [])
            metadata = payload.get("metadata", {})
            if not isinstance(episodes, list) or not isinstance(generations, list) or not isinstance(metadata, dict):
                raise ValueError("history fields have invalid types")
            self._metadata.update(metadata)
            neat = metadata.get("neat", {})
            current_generation = neat.get("current_generation") if isinstance(neat, dict) else None
            if current_generation is None:
                self._episodes.extend(item for item in episodes if isinstance(item, dict))
            else:
                self._episodes.extend(
                    item for item in episodes if isinstance(item, dict) and item.get("generation") == current_generation
                )
            self._generations.extend(item for item in generations if isinstance(item, dict))
            self._sequence = max(
                (item.get("index", 0) for item in self._episodes if isinstance(item.get("index"), int)),
                default=0,
            )
            self._started_at = payload.get("started_at") or self._started_at
            logger.info(f"Loaded {len(self._episodes)} training episodes from Redis key {self._history_key}")
        except (RedisError, ValueError, json.JSONDecodeError) as error:
            logger.warning(f"Ignoring unreadable training UI history in Redis key {self._history_key}: {error}")

    def _mark_dirty(self) -> None:
        if self._history_key is not None:
            self._persist_event.set()

    def _persist_loop(self) -> None:
        while True:
            self._persist_event.wait()
            with self._lock:
                version = self._history_version
            time.sleep(0.25)
            with self._lock:
                if version == self._history_version:
                    self._persist_event.clear()
            self.flush(expected_version=version)


_store = TelemetryStore()


def get_store() -> TelemetryStore:
    return _store


def configure_history(
    scope: str,
    *,
    redis_url: str = "redis://127.0.0.1:6379/0",
    key_prefix: str = "datenwissenschaften:history",
) -> None:
    _store.configure_history(scope, redis_url=redis_url, key_prefix=key_prefix)


def clear_metadata(section: str, *, clear_history: bool = False) -> None:
    _store.clear_metadata(section, clear_history=clear_history)


def publish_episode(**values: Any) -> None:
    _store.publish_episode(values)


def publish_generation(**values: Any) -> None:
    _store.publish_generation(values)


def publish_metadata(section: str, values: dict[str, Any], *, replace: bool = False) -> None:
    _store.publish_metadata(section, values, replace=replace)


atexit.register(_store.flush)
