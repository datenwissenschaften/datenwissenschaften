from __future__ import annotations

import json
from typing import Any

from redis import Redis
from redis.exceptions import RedisError


class RedisStore:
    """Small namespaced Redis store for non-model training state."""

    def __init__(self, redis_url: str, *, key_prefix: str = "datenwissenschaften") -> None:
        self._redis = Redis.from_url(redis_url, decode_responses=False)
        self._prefix = key_prefix.rstrip(":")
        try:
            self._redis.ping()
        except RedisError as error:
            raise RuntimeError(f"Could not connect to Redis store at {redis_url}: {error}") from error

    def get(self, *parts: str, default: Any = None) -> Any:
        value = self._redis.get(self.key(*parts))
        if value is None:
            return default
        return json.loads(value)

    def set(self, *parts: str, value: Any) -> None:
        self._redis.set(self.key(*parts), json.dumps(value, separators=(",", ":")))

    def delete(self, *parts: str) -> None:
        self._redis.delete(self.key(*parts))

    def delete_prefix(self, *parts: str) -> None:
        """Delete a namespace and all keys nested below it."""
        prefix = self.key(*parts)
        keys = list(self._redis.scan_iter(match=f"{prefix}:*"))
        if keys:
            self._redis.delete(*keys)
        self._redis.delete(prefix)

    def key(self, *parts: str) -> str:
        escaped = (str(part).replace(":", "_") for part in parts)
        return ":".join((self._prefix, *escaped))
