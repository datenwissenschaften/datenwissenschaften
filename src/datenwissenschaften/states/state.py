from pathlib import Path
from typing import Generic, TypeVar

import numpy as np

from datenwissenschaften.ram.model import RamInfo

T = TypeVar("T", bound=RamInfo)


class State(Generic[T]):
    ram: T
    frame: np.ndarray
    model_dir: Path

    def __init__(self, model_dir: Path) -> None:
        self._validate_type()
        self.model_dir = model_dir

    def reset(self, ram: T, frame: np.ndarray) -> None:
        self.ram = ram
        self.frame = frame
        self._detect()
        self._on_reset()

    def step(
        self,
        ram: T,
        frame: np.ndarray,
    ) -> tuple[float, bool, bool, type["State[T]"] | None]:
        self.ram = ram
        self.frame = frame
        self._detect()
        self._on_step()
        return self._automatic_reward(), self._terminated(), self._truncated(), self._next()

    def _detect(self) -> None:
        pass

    def _on_reset(self) -> None:
        pass

    def _on_step(self) -> None:
        pass

    def _automatic_reward(self) -> float:
        raise NotImplementedError

    def _terminated(self) -> bool:
        return False

    def _truncated(self) -> bool:
        return False

    def _won(self) -> bool:
        return False

    def _next(self) -> type["State[T]"] | None:
        return None

    def _validate_type(self) -> None:
        from datenwissenschaften.states.explorer import Explorer
        from datenwissenschaften.states.ram_scorer import RamScorerState
        from datenwissenschaften.states.target import TargetState

        state_type = type(self)
        if not isinstance(self, (TargetState, RamScorerState)):
            raise TypeError(f"{state_type.__name__} must inherit Explorer, TargetState or RamScorerState")
        if state_type.step is not State.step:
            raise TypeError(f"{state_type.__name__} cannot override step")
        if state_type._automatic_reward not in {
            Explorer._automatic_reward,
            TargetState._automatic_reward,
            RamScorerState._automatic_reward,
        }:
            raise TypeError(f"{state_type.__name__} cannot define custom rewards")
        if any(
            "_reward" in parent.__dict__
            for parent in state_type.__mro__
            if parent not in {Explorer, TargetState, RamScorerState}
        ):
            raise TypeError(f"{state_type.__name__} cannot define custom rewards")
        if isinstance(self, Explorer):
            if state_type._target_state is Explorer._target_state:
                raise TypeError(f"{state_type.__name__} must define a target state")
            if state_type._won is not Explorer._won:
                raise TypeError(f"{state_type.__name__} cannot define won")
            return

        if isinstance(self, RamScorerState):
            if state_type._scored_value is RamScorerState._scored_value:
                raise TypeError(f"{state_type.__name__} must define a scored value")

        has_next = state_type._next is not State._next
        has_won = state_type._won is not State._won
        if has_next == has_won:
            raise TypeError(f"{state_type.__name__} must define exactly one of next or won")
