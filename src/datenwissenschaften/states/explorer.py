from abc import ABC, abstractmethod
from typing import TypeVar

from datenwissenschaften.ram import RamInfo
from datenwissenschaften.states.state import State
from datenwissenschaften.states.target import TargetState

T = TypeVar("T", bound=RamInfo)


class Explorer(TargetState[T], ABC):
    description = ""

    area_discovery_reward = 100.0
    position_discovery_reward = 1.0
    target_found_reward = 10000.0

    visited_areas: set[tuple[int, int]]
    visited_positions: set[tuple[int, int]]

    def _on_reset(self) -> None:
        position = self._actor_position(self.ram)
        self.visited_areas = {position.screen}
        self.visited_positions = {position.coordinates}
        super()._on_reset()

    def _target_reward(self, distance: float | None) -> float:
        position = self._actor_position(self.ram)
        current_area = position.screen
        area_is_unseen = current_area not in self.visited_areas
        self.visited_areas.add(current_area)

        current_position = position.coordinates
        position_is_unseen = current_position not in self.visited_positions
        self.visited_positions.add(current_position)

        reward = super()._target_reward(distance)
        if area_is_unseen:
            reward += self.area_discovery_reward
        if position_is_unseen:
            reward += self.position_discovery_reward
        if distance is not None:
            self.remember_detected_target()
            reward += self.target_found_reward
        return reward

    def _next(self) -> type[State[T]] | None:
        return self._target_state() if self.target_detector.seen else None

    def _won(self) -> bool:
        return False

    @abstractmethod
    def _target_state(self) -> type[State[T]]:
        pass
