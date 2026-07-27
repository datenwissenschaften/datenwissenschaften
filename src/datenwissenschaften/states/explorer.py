from typing import TypeVar

from datenwissenschaften.ram.model import RamInfo
from datenwissenschaften.states.state import State
from datenwissenschaften.states.target import TargetState

T = TypeVar("T", bound=RamInfo)


class Explorer(TargetState[T]):
    visits: dict[tuple[int, int, int, int], int]

    def _on_reset(self) -> None:
        self.visits = {}
        super()._on_reset()

    def _reward(self) -> float:
        location = (
            int(self.ram.screen_x),
            int(self.ram.screen_y),
            int(self.ram.position_x),
            int(self.ram.position_y),
        )
        visits = self.visits.get(location, 0) + 1
        self.visits[location] = visits
        return super()._reward() + float(visits == 1)

    def _next(self) -> type[State[T]] | None:
        if self.target_detector is not None and self.target_detector.seen:
            return self._target_state()
        return None

    def _won(self) -> bool:
        return False

    def _target_state(self) -> type[State[T]]:
        raise NotImplementedError
