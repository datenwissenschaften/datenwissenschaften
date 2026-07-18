from abc import ABC, abstractmethod
from typing import TypeVar

import numpy as np

from datenwissenschaften.ram import RamInfo
from datenwissenschaften.states.state import State
from datenwissenschaften.states.target import TargetState

T = TypeVar("T", bound=RamInfo)


class Explorer(TargetState[T], ABC):
    description = ""

    # The target is normally off-screen during exploration. Penalizing every
    # absent frame makes short, failed episodes preferable to searching.
    target_missing_penalty = 0.0
    area_discovery_reward = 100.0
    position_discovery_reward = 1.0
    position_grid_size = 8
    horizontal_frontier_reward_scale = 1.0
    vertical_frontier_reward_scale = 0.10
    maximum_frontier_reward = 16.0
    frontier_stall_grace_steps = 20
    frontier_stall_penalty_scale = 0.05
    maximum_frontier_stall_penalty = 2.0
    frontier_staleness_limit = 600
    target_found_reward = 10000.0

    visited_areas: set[tuple[int, int]]
    visited_positions: set[tuple[int, int]]
    frontier_min_x: int
    frontier_max_x: int
    frontier_min_y: int
    frontier_max_y: int
    steps_since_frontier: int

    def _on_reset(self) -> None:
        position = self._actor_position(self.ram)
        self.visited_areas = {position.screen}
        self.visited_positions = {self._position_bucket(position.coordinates)}
        self.frontier_min_x = self.frontier_max_x = position.x
        self.frontier_min_y = self.frontier_max_y = position.y
        self.steps_since_frontier = 0
        super()._on_reset()

    def _target_reward(self, distance: float | None) -> float:
        position = self._actor_position(self.ram)
        current_area = position.screen
        area_is_unseen = current_area not in self.visited_areas
        self.visited_areas.add(current_area)

        current_position = self._position_bucket(position.coordinates)
        position_is_unseen = current_position not in self.visited_positions
        self.visited_positions.add(current_position)

        reward = super()._target_reward(distance)
        reward += self._frontier_reward(position.coordinates)
        if area_is_unseen:
            reward += self.area_discovery_reward
        if position_is_unseen:
            reward += self.position_discovery_reward
        if distance is not None:
            self.remember_detected_target()
            reward += self.target_found_reward
        return reward

    def auxiliary_features(self, ram: T | None = None) -> list[float]:
        return super().auxiliary_features(ram) + self._exploration_features(ram)

    def _frontier_reward(self, coordinates: tuple[int, int]) -> float:
        """Reward expanding the explored map, not merely revisiting pixels."""
        x, y = coordinates
        horizontal_expansion = max(0, self.frontier_min_x - x) + max(0, x - self.frontier_max_x)
        vertical_expansion = max(0, self.frontier_min_y - y) + max(0, y - self.frontier_max_y)
        self.frontier_min_x = min(self.frontier_min_x, x)
        self.frontier_max_x = max(self.frontier_max_x, x)
        self.frontier_min_y = min(self.frontier_min_y, y)
        self.frontier_max_y = max(self.frontier_max_y, y)

        expansion_reward = (
            horizontal_expansion * self.horizontal_frontier_reward_scale
            + vertical_expansion * self.vertical_frontier_reward_scale
        )
        if expansion_reward > 0.0:
            self.steps_since_frontier = 0
            return min(self.maximum_frontier_reward, expansion_reward)

        self.steps_since_frontier += 1
        stalled_steps = max(0, self.steps_since_frontier - self.frontier_stall_grace_steps)
        return -min(
            self.maximum_frontier_stall_penalty,
            stalled_steps * self.frontier_stall_penalty_scale,
        )

    def _exploration_features(self, ram: T | None) -> list[float]:
        if ram is None or not hasattr(self, "frontier_min_x"):
            return [0.0] * 5
        position = self._actor_position(ram)
        scale = max(1.0, float(position.screen_size))
        return [
            float(np.clip((position.x - self.frontier_min_x) / scale, 0.0, 1.0)),
            float(np.clip((self.frontier_max_x - position.x) / scale, 0.0, 1.0)),
            float(np.clip((position.y - self.frontier_min_y) / scale, 0.0, 1.0)),
            float(np.clip((self.frontier_max_y - position.y) / scale, 0.0, 1.0)),
            float(np.clip(self.steps_since_frontier / max(1, self.frontier_stall_grace_steps), 0.0, 1.0)),
        ]

    def _position_bucket(self, coordinates: tuple[int, int]) -> tuple[int, int]:
        grid_size = max(1, self.position_grid_size)
        return coordinates[0] // grid_size, coordinates[1] // grid_size

    def _truncated(self) -> bool:
        """End an attempt that has stopped expanding its exploration frontier."""
        limit = max(0, int(self.frontier_staleness_limit))
        target_seen = bool(getattr(self.target_detector, "seen", False))
        return limit > 0 and self.steps_since_frontier >= limit and not target_seen

    def _next(self) -> type[State[T]] | None:
        return self._target_state() if self.target_detector.seen else None

    def _won(self) -> bool:
        return False

    @abstractmethod
    def _target_state(self) -> type[State[T]]:
        pass
