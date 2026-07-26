from pathlib import Path
from typing import Generic, TypeVar

import numpy as np

from datenwissenschaften.ram.model import RamInfo
from datenwissenschaften.states.detector import TemplateDetector

T = TypeVar("T", bound=RamInfo)


class State(Generic[T]):
    template_file: str
    target_detector: TemplateDetector | None
    ram: T
    frame: np.ndarray
    observation: np.ndarray
    model_dir: Path

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        self.target_detector = TemplateDetector(self.template_file) if hasattr(self, "template_file") else None

    def reset(self, ram: T, frame: np.ndarray, observation: np.ndarray) -> None:
        self.ram = ram
        self.frame = frame
        self.observation = observation
        self._on_reset()

    def step(
        self,
        ram: T,
        frame: np.ndarray,
        observation: np.ndarray,
    ) -> tuple[float, bool, bool, type["State[T]"] | None]:
        self.ram = ram
        self.frame = frame
        self.observation = observation
        return self._reward(), self._terminated(), self._truncated(), self._next()

    def _on_reset(self) -> None:
        pass

    def _reward(self) -> float:
        return 0.0

    def _terminated(self) -> bool:
        return False

    def _truncated(self) -> bool:
        return False

    def _won(self) -> bool:
        return False

    def _next(self) -> type["State[T]"] | None:
        return None
