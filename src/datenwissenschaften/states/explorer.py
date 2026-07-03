from abc import ABC, abstractmethod
from collections.abc import Hashable
from typing import TypeVar

from datenwissenschaften.ram import RamInfo
from datenwissenschaften.states.state import State
from datenwissenschaften.states.target import TargetState

T = TypeVar("T", bound=RamInfo)


class Explorer(TargetState[T], ABC):
    description = ""
    progress = -1

    area_discovery_reward = 100.0
    position_discovery_reward = 1.0
    target_found_reward = 100.0

    visited_areas: set[Hashable]
    visited_positions: set[Hashable]

    def _on_reset(self) -> None:
        self.visited_areas = {self._area_key(self.ram)}
        self.visited_positions = {self._position_key(self.ram)}
        super()._on_reset()

    def _target_reward(self, distance: float | None) -> float:
        current_area = self._area_key(self.ram)
        area_is_unseen = current_area not in self.visited_areas
        self.visited_areas.add(current_area)

        current_position = self._position_key(self.ram)
        position_is_unseen = current_position not in self.visited_positions
        self.visited_positions.add(current_position)

        reward = self.area_discovery_reward if area_is_unseen else 0.0
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
    def _area_key(self, ram: T) -> Hashable:
        pass

    @abstractmethod
    def _position_key(self, ram: T) -> Hashable:
        pass

    @abstractmethod
    def _target_state(self) -> type[State[T]]:
        pass
