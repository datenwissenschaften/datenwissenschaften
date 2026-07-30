from typing import TypeVar

from datenwissenschaften.ram.model import RamInfo
from datenwissenschaften.states.state import State
from datenwissenschaften.states.target import TargetState

T = TypeVar("T", bound=RamInfo)
EXPLORATION_REWARD = 0.1
EXPLORATION_TILE_SIZE = 8


class Explorer(TargetState[T]):
    visits: dict[tuple[int, int, int, int], int]

    def _on_reset(self) -> None:
        self.visits = {}
        super()._on_reset()

    def _automatic_reward(self) -> float:
        location = (
            int(self.ram.screen_x),
            int(self.ram.screen_y),
            int(self.ram.player_x) // EXPLORATION_TILE_SIZE,
            int(self.ram.player_y) // EXPLORATION_TILE_SIZE,
        )
        visits = self.visits.get(location, 0) + 1
        self.visits[location] = visits
        return super()._automatic_reward() + EXPLORATION_REWARD * float(visits == 1)

    def _next(self) -> type[State[T]] | None:
        if self.target_detector.seen:
            return self._target_state()
        return None

    def _won(self) -> bool:
        return False

    def _target_state(self) -> type[State[T]]:
        raise NotImplementedError
