from typing import TypeVar

from datenwissenschaften.ram.model import RamInfo
from datenwissenschaften.states.target import TargetState

T = TypeVar("T", bound=RamInfo)


class RamScorerState(TargetState[T]):
    previous_value: float | None = None

    def _on_reset(self) -> None:
        super()._on_reset()
        self.previous_value = float(self._scored_value())

    def _automatic_reward(self) -> float:
        current_value = float(self._scored_value())
        previous = self.previous_value
        self.previous_value = current_value

        if previous is None:
            return super()._automatic_reward()

        reward = current_value - previous
        return super()._automatic_reward() + reward

    def _scored_value(self) -> float:
        raise NotImplementedError
