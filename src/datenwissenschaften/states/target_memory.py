import json
import os
from pathlib import Path

from loguru import logger


class TargetMemory:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.coordinates = self._load()

    def remember(self, coordinates: tuple[float, float]) -> None:
        if self.coordinates is not None:
            return
        self.coordinates = coordinates
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(coordinates), encoding="utf-8")
        temporary.replace(self.path)
        logger.success(
            "Remembered {} target at ({:.1f}, {:.1f})",
            self.path.stem,
            coordinates[0],
            coordinates[1],
        )

    def _load(self) -> tuple[float, float] | None:
        if not self.path.is_file():
            return None
        value = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(value, list) or len(value) != 2 or any(not isinstance(item, int | float) for item in value):
            raise RuntimeError(f"Invalid target memory: {self.path}")
        return float(value[0]), float(value[1])
