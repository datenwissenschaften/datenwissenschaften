from math import tanh
from typing import Sequence

from datenwissenschaften.persistence import RedisStore
from datenwissenschaften.settings import load_config


class TargetMemory:
    _registry: dict[str, "TargetMemory"] = {}

    def __init__(
        self,
        key: str,
        *,
        origin: Sequence[float],
        scale: float | Sequence[float] = 1.0,
    ) -> None:
        config = load_config()
        self.key = key
        self._store = RedisStore(config.ui.redis_url)
        self._scope = (
            "target-memory",
            config.training.game_identity,
            config.training.active_savestate or "default",
            key,
        )
        self.origin = self._coordinates(origin)
        self.scale = self._scale(scale, len(self.origin))
        self.coordinates = self._load()

    @classmethod
    def shared(
        cls,
        key: str,
        *,
        origin: Sequence[float],
        scale: float | Sequence[float] = 1.0,
    ) -> "TargetMemory":
        if key not in cls._registry:
            cls._registry[key] = cls(key, origin=origin, scale=scale)

        memory = cls._registry[key]
        expected_origin = cls._coordinates(origin)
        expected_scale = cls._scale(scale, len(expected_origin))
        if memory.origin != expected_origin or memory.scale != expected_scale:
            raise ValueError(f"Target memory {key} was requested with an incompatible coordinate schema")
        return memory

    def remember(self, coordinates: Sequence[float]) -> bool:
        if self.coordinates is not None:
            return False

        self.coordinates = self._validate_dimensions(coordinates)
        self._save()
        return True

    def features(self, current_coordinates: Sequence[float] | None = None) -> list[float]:
        current = self.origin if current_coordinates is None else self._validate_dimensions(current_coordinates)
        if self.coordinates is None:
            deltas = (0.0 for _ in self.origin)
            known = 0.0
        else:
            deltas = (
                (target_coordinate - current_coordinate) / coordinate_scale
                for target_coordinate, current_coordinate, coordinate_scale in zip(
                    self.coordinates,
                    current,
                    self.scale,
                    strict=True,
                )
            )
            known = 1.0
        return [known, *(tanh(delta) for delta in deltas)]

    def _validate_dimensions(self, coordinates: Sequence[float]) -> tuple[float, ...]:
        values = self._coordinates(coordinates)
        if len(values) != len(self.origin):
            raise ValueError(f"Expected {len(self.origin)} target coordinates, received {len(values)}")
        return values

    def _load(self) -> tuple[float, ...] | None:
        try:
            payload = self._store.get(*self._scope)
            return self._validate_dimensions(payload["coordinates"])
        except (KeyError, TypeError, ValueError):
            return None

    def _save(self) -> None:
        if self.coordinates is None:
            return

        self._store.set(*self._scope, value={"coordinates": self.coordinates})

    @staticmethod
    def _coordinates(values: Sequence[float]) -> tuple[float, ...]:
        coordinates = tuple(float(value) for value in values)
        if not coordinates:
            raise ValueError("Target memory requires at least one coordinate dimension")
        return coordinates

    @staticmethod
    def _scale(value: float | Sequence[float], dimensions: int) -> tuple[float, ...]:
        if isinstance(value, (int, float)):
            scales = (float(value),) * dimensions
        else:
            scales = tuple(float(item) for item in value)
        if len(scales) != dimensions or any(item <= 0 for item in scales):
            raise ValueError("Target memory scale must contain one positive value per coordinate dimension")
        return scales
