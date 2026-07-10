from collections.abc import Callable
from typing import Generic, TypeVar

import numpy as np

from datenwissenschaften.ram import RamInfo
from datenwissenschaften.states.state import State

T = TypeVar("T", bound=RamInfo)

TransitionListener = Callable[[str, str], None]


class StateMachine(Generic[T]):
    def __init__(
        self,
        start_state: State[T],
        *,
        terminate_on_transition: bool,
        transition_bonus: float,
        on_transition: TransitionListener,
    ):
        self.start_state = start_state
        self.current_state = start_state
        self.terminate_on_transition = terminate_on_transition
        self.transition_bonus = transition_bonus
        self.on_transition = on_transition
        self.last_transition: tuple[str, str] | None = None
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
        self.last_transition = None
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

        self.last_transition = None
        if next_state_type is not None:
            previous_state_name = self.state_name
            self.current_state = self._get_or_create_state(next_state_type)
            self.current_state.reset(ram, frame, observation)
            self.last_transition = (previous_state_name, self.state_name)
            reward += self.transition_bonus
            self.on_transition(previous_state_name, self.state_name)
            if self.terminate_on_transition:
                terminated = True

        return reward, terminated, truncated

    def features(self) -> list[float]:
        return self.current_state.features()

    @property
    def state_name(self) -> str:
        return self.current_state.__class__.__name__

    def _get_or_create_state(self, state_cls: type[State[T]]) -> State[T]:
        if state_cls not in self.states_by_type:
            self.states_by_type[state_cls] = state_cls()

        return self.states_by_type[state_cls]
