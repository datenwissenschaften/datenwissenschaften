from typing import Generic, TypeVar

import numpy as np

from datenwissenschaften.ram.model import RamInfo
from datenwissenschaften.states.state import State

T = TypeVar("T", bound=RamInfo)


class StateMachine(Generic[T]):
    def __init__(self, start: State[T]) -> None:
        self.start = start
        self.current = start
        self._states: dict[type[State[T]], State[T]] = {type(start): start}

    @property
    def name(self) -> str:
        return type(self.current).__name__

    def reset(
        self,
        ram: T,
        frame: np.ndarray,
        state_type: type[State[T]],
    ) -> None:
        if state_type not in self._states:
            self._states[state_type] = state_type(self.start.model_dir)
        self.current = self._states[state_type]
        self.current.reset(ram, frame)

    def step(self, ram: T, frame: np.ndarray) -> tuple[float, bool, bool]:
        reward, terminated, truncated, next_type = self.current.step(ram, frame)
        if next_type is not None:
            if next_type not in self._states:
                self._states[next_type] = next_type(self.start.model_dir)
            self.current = self._states[next_type]
            self.current.reset(ram, frame)
        return reward, terminated, truncated
