from typing import TypeVar

from datenwissenschaften.ram.model import RamInfo
from datenwissenschaften.states.state import State

T = TypeVar("T", bound=RamInfo)


class RamScorerState(State[T]):
    previous_value: float | None = None

    def _on_reset(self) -> None:
        super()._on_reset()
        self.previous_value = float(self._scored_value())

    def _automatic_reward(self) -> float:
        current_value = float(self._scored_value())
        previous = self.previous_value
        self.previous_value = current_value

        if previous is None:
            return 0.0

        return current_value - previous

    def _scored_value(self) -> float:
        raise NotImplementedError
