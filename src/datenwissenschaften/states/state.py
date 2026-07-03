from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import numpy as np

from datenwissenschaften.ram import RamInfo
from datenwissenschaften.vision.hybrid_encoder import HybridEncoder

T = TypeVar("T", bound=RamInfo)


class State(ABC, Generic[T]):
    description = ""
    visual_encoder = HybridEncoder()

    ram: T
    frame: np.ndarray
    observation: np.ndarray
    savestate: bytes | None = None
    progress: int
    beaten: bool = False

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if "progress" not in cls.__dict__ or not isinstance(cls.__dict__["progress"], int):
            raise TypeError(f"{cls.__name__} must define an integer progress value.")

    def save(self, savestate: bytes) -> bool:
        if self.beaten or self.savestate is not None:
            return False

        self.savestate = bytes(savestate)
        return True

    def load(self, savestate: bytes) -> None:
        self.savestate = bytes(savestate)

    def delete_savestate(self) -> bool:
        existed = self.savestate is not None
        self.savestate = None
        return existed

    def clear_saved_progress(self) -> None:
        self.savestate = None
        self.beaten = False

    def mark_beaten(self) -> bool:
        changed = not self.beaten
        self.beaten = True
        self.savestate = None
        return changed

    def reset(
        self,
        ram: T,
        frame: np.ndarray,
        observation: np.ndarray,
    ) -> None:
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

        reward = self._reward()
        terminated = self._terminated()
        truncated = self._truncated()
        next_state = self._next()

        return reward, terminated, truncated, next_state

    def features(self) -> list[float]:
        return self.visual_encoder.encode(self.observation, self.ram) + self.auxiliary_features(self.ram)

    def auxiliary_features(self, ram: T | None = None) -> list[float]:
        return []

    def _on_reset(self) -> None:
        pass

    @abstractmethod
    def _reward(self) -> float:
        pass

    @abstractmethod
    def _terminated(self) -> bool:
        pass

    @abstractmethod
    def _truncated(self) -> bool:
        pass

    @abstractmethod
    def _won(self) -> bool:
        return False

    def _next(self) -> type["State[T]"] | None:
        return None
