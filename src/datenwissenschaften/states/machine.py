from typing import Generic, TypeVar

import numpy as np

from datenwissenschaften.ram import RamInfo
from datenwissenschaften.states.state import State

T = TypeVar("T", bound=RamInfo)


class StateMachine(Generic[T]):
    def __init__(self, start_state: State[T]):
        self.start_state = start_state
        self.current_state = start_state
        self.states_by_type: dict[type[State[T]], State[T]] = {
            type(start_state): start_state,
        }

    def reset(
        self,
        ram: T,
        frame: np.ndarray,
        observation: np.ndarray,
        state_type: type[State[T]] | None = None,
    ) -> None:
        self.current_state = self._get_or_create_state(state_type) if state_type is not None else self.start_state
        self.current_state.reset(ram, frame, observation)

    def step(
        self,
        ram: T,
        frame: np.ndarray,
        observation: np.ndarray,
    ) -> tuple[float, bool, bool]:
        reward, terminated, truncated, next_state_type = self.current_state.step(
            ram,
            frame,
            observation,
        )

        if next_state_type is not None:
            self.current_state = self._get_or_create_state(next_state_type)
            self.current_state.reset(ram, frame, observation)

        return reward, terminated, truncated

    def features(self) -> list[float]:
        return self.current_state.features()

    @property
    def state_name(self) -> str:
        return self.current_state.__class__.__name__

    def savestate(self, state_type: type[State[T]]) -> bytes | None:
        return self._get_or_create_state(state_type).savestate

    def save_current_state(self, savestate: bytes) -> bool:
        return self.current_state.save(savestate)

    def load_savestate(self, state_type: type[State[T]], savestate: bytes) -> None:
        self._get_or_create_state(state_type).load(savestate)

    def delete_savestate(self, state_type: type[State[T]]) -> bool:
        return self._get_or_create_state(state_type).delete_savestate()

    def mark_beaten(self, state_type: type[State[T]]) -> bool:
        return self._get_or_create_state(state_type).mark_beaten()

    def is_beaten(self, state_type: type[State[T]]) -> bool:
        return self._get_or_create_state(state_type).beaten

    def _get_or_create_state(self, state_cls: type[State[T]]) -> State[T]:
        if state_cls not in self.states_by_type:
            self.states_by_type[state_cls] = state_cls()

        return self.states_by_type[state_cls]
