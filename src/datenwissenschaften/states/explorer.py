from abc import ABC, abstractmethod
from math import log1p
from typing import TypeVar

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
    region_discovery_reward = 8.0
    position_discovery_reward = 1.0
    region_grid_size = 32
    position_grid_size = 8
    revisit_penalty_scale = 0.05
    maximum_revisit_penalty = 0.25
    revisit_penalty_grace_steps = 20
    horizontal_frontier_reward_scale = 1.0
    vertical_frontier_reward_scale = 0.10
    maximum_frontier_reward = 16.0
    # A larger discontinuity is probably a transient RAM value, death, or
    # teleport. Do not let it permanently poison the episode's frontier.
    maximum_coordinate_jump_screens = 1.0
    frontier_staleness_limit = 600
    target_found_reward = 10000.0

    visited_areas: set[tuple[int, int]]
    visited_regions: set[tuple[int, int]]
    visited_positions: set[tuple[int, int]]
    position_visit_counts: dict[tuple[int, int], int]
    frontier_min_x: int
    frontier_max_x: int
    frontier_min_y: int
    frontier_max_y: int
    steps_since_frontier: int
    previous_coordinates: tuple[int, int]
    invalid_position_steps: int

    def _on_reset(self) -> None:
        position = self._actor_position(self.ram)
        position_bucket = self._position_bucket(position.coordinates)
        self.visited_areas = {position.screen}
        self.visited_regions = {self._region_bucket(position.coordinates)}
        self.visited_positions = {position_bucket}
        self.position_visit_counts = {position_bucket: 1}
        self.frontier_min_x = self.frontier_max_x = position.x
        self.frontier_min_y = self.frontier_max_y = position.y
        self.steps_since_frontier = 0
        self.previous_coordinates = position.coordinates
        self.invalid_position_steps = 0
        super()._on_reset()

    def _target_reward(self, distance: float | None) -> float:
        position = self._actor_position(self.ram)
        reward = super()._target_reward(distance)
        reward += self._exploration_reward(
            position.screen,
            position.coordinates,
            screen_size=position.screen_size,
        )
        if distance is not None:
            self.remember_detected_target()
            reward += self.target_found_reward
        return reward

    def _exploration_reward(
        self,
        screen: tuple[int, int],
        coordinates: tuple[int, int],
        *,
        screen_size: int = 256,
    ) -> float:
        """Score coverage with bounded pressure against repeatedly stalling."""
        if not self._position_is_plausible(coordinates, screen_size):
            self.invalid_position_steps += 1
            self.steps_since_frontier += 1
            return -self._adaptive_revisit_penalty(1)

        self.previous_coordinates = coordinates

        area_is_unseen = screen not in self.visited_areas
        if area_is_unseen:
            self.visited_areas.add(screen)

        current_region = self._region_bucket(coordinates)
        region_is_unseen = current_region not in self.visited_regions
        if region_is_unseen:
            self.visited_regions.add(current_region)

        current_position = self._position_bucket(coordinates)
        previous_visits = self.position_visit_counts.get(current_position, 0)
        position_is_unseen = previous_visits == 0
        self.position_visit_counts[current_position] = previous_visits + 1
        if position_is_unseen:
            self.visited_positions.add(current_position)

        frontier_reward = self._frontier_reward(coordinates)
        discovered = area_is_unseen or region_is_unseen or position_is_unseen
        if discovered:
            # Novel coverage inside the existing outer bounds is meaningful
            # exploration too, so it resets the stale-attempt clock.
            self.steps_since_frontier = 0
            frontier_reward = max(0.0, frontier_reward)

        reward = frontier_reward
        if area_is_unseen:
            reward += self.area_discovery_reward
        if region_is_unseen:
            reward += self.region_discovery_reward
        if position_is_unseen:
            reward += self.position_discovery_reward
        else:
            reward -= self._adaptive_revisit_penalty(previous_visits)
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
        return 0.0

    def _adaptive_revisit_penalty(self, previous_visits: int) -> float:
        """Increase loop pressure smoothly as an attempt approaches timeout."""
        grace = max(0, int(self.revisit_penalty_grace_steps))
        limit = int(self.frontier_staleness_limit)
        if limit <= grace:
            return 0.0
        stale_progress = min(1.0, max(0.0, (self.steps_since_frontier - grace) / (limit - grace)))
        visit_pressure = log1p(max(0, previous_visits))
        return min(
            max(0.0, self.maximum_revisit_penalty),
            max(0.0, self.revisit_penalty_scale) * visit_pressure * stale_progress**2,
        )

    def _position_is_plausible(self, coordinates: tuple[int, int], screen_size: int) -> bool:
        allowed_screens = float(self.maximum_coordinate_jump_screens)
        if allowed_screens <= 0.0:
            return True
        maximum_jump = max(1.0, float(screen_size) * allowed_screens)
        return (
            abs(coordinates[0] - self.previous_coordinates[0]) <= maximum_jump
            and abs(coordinates[1] - self.previous_coordinates[1]) <= maximum_jump
        )

    def _exploration_features(self, ram: T | None) -> list[float]:
        if ram is None or not hasattr(self, "frontier_min_x"):
            return [0.0] * 5
        position = self._actor_position(ram)
        scale = max(1.0, float(position.screen_size))
        return [
            _unit_interval((position.x - self.frontier_min_x) / scale),
            _unit_interval((self.frontier_max_x - position.x) / scale),
            _unit_interval((position.y - self.frontier_min_y) / scale),
            _unit_interval((self.frontier_max_y - position.y) / scale),
            (
                _unit_interval(self.steps_since_frontier / self.frontier_staleness_limit)
                if self.frontier_staleness_limit > 0
                else 0.0
            ),
        ]

    def _position_bucket(self, coordinates: tuple[int, int]) -> tuple[int, int]:
        grid_size = max(1, self.position_grid_size)
        return coordinates[0] // grid_size, coordinates[1] // grid_size

    def _region_bucket(self, coordinates: tuple[int, int]) -> tuple[int, int]:
        grid_size = max(1, self.region_grid_size)
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


def _unit_interval(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
