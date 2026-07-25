from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import isqrt, log1p, sqrt
from typing import Self, TypeVar

from datenwissenschaften.helpers.position import Position
from datenwissenschaften.ram import RamInfo
from datenwissenschaften.states.state import State
from datenwissenschaften.states.target import TargetState

T = TypeVar("T", bound=RamInfo)
Coordinates = tuple[int, int]


@dataclass(frozen=True, slots=True)
class ExplorationSettings:
    screen_size: int
    step_penalty: float

    @property
    def position_grid_size(self) -> int:
        return max(1, isqrt(self.screen_size))

    @property
    def region_grid_size(self) -> int:
        return max(self.position_grid_size, isqrt(self.screen_size * self.position_grid_size))

    @property
    def position_reward(self) -> float:
        return self.position_grid_size / self.screen_size

    @property
    def region_reward(self) -> float:
        return self.region_grid_size / self.screen_size

    @property
    def area_reward(self) -> float:
        return 1.0

    @property
    def revisit_penalty_grace_steps(self) -> int:
        return max(1, self.screen_size // self.position_grid_size)

    @property
    def staleness_limit(self) -> int:
        return self.revisit_penalty_grace_steps**2

    @classmethod
    def create(cls, screen_size: int, step_penalty: float) -> Self:
        if screen_size <= 0:
            raise ValueError("Screen size must be positive")
        return cls(screen_size, max(0.0, step_penalty))


@dataclass(frozen=True, slots=True)
class Discovery:
    area: bool
    region: bool
    position: bool
    previous_visits: int

    @property
    def found(self) -> bool:
        return self.area or self.region or self.position


@dataclass(slots=True)
class Coverage:
    areas: set[Coordinates]
    regions: set[Coordinates]
    position_visits: dict[Coordinates, int]
    region_grid_size: int
    position_grid_size: int

    @classmethod
    def start(cls, screen: Coordinates, coordinates: Coordinates, settings: ExplorationSettings) -> Self:
        region = cls.bucket(coordinates, settings.region_grid_size)
        position = cls.bucket(coordinates, settings.position_grid_size)
        return cls(
            {screen},
            {region},
            {position: 1},
            settings.region_grid_size,
            settings.position_grid_size,
        )

    def discover(self, screen: Coordinates, coordinates: Coordinates) -> Discovery:
        area_is_new = screen not in self.areas
        region = self.bucket(coordinates, self.region_grid_size)
        region_is_new = region not in self.regions
        position = self.bucket(coordinates, self.position_grid_size)
        previous_visits = self.position_visits.get(position, 0)
        position_is_new = previous_visits == 0
        self.areas.add(screen)
        self.regions.add(region)
        self.position_visits[position] = previous_visits + 1
        return Discovery(area_is_new, region_is_new, position_is_new, previous_visits)

    @staticmethod
    def bucket(coordinates: Coordinates, grid_size: int) -> Coordinates:
        size = max(1, grid_size)
        return coordinates[0] // size, coordinates[1] // size


@dataclass(slots=True)
class Frontier:
    minimum_x: int
    maximum_x: int
    minimum_y: int
    maximum_y: int
    previous_coordinates: Coordinates
    stale_steps: int

    @classmethod
    def start(cls, coordinates: Coordinates) -> Self:
        x, y = coordinates
        return cls(x, x, y, y, coordinates, 0)

    def expand(self, coordinates: Coordinates, settings: ExplorationSettings) -> float:
        x, y = coordinates
        horizontal = max(0, self.minimum_x - x) + max(0, x - self.maximum_x)
        vertical = max(0, self.minimum_y - y) + max(0, y - self.maximum_y)
        self.minimum_x = min(self.minimum_x, x)
        self.maximum_x = max(self.maximum_x, x)
        self.minimum_y = min(self.minimum_y, y)
        self.maximum_y = max(self.maximum_y, y)
        reward = (horizontal + vertical) / settings.screen_size
        if reward <= 0.0:
            self.stale_steps += 1
            return 0.0
        self.stale_steps = 0
        return min(settings.area_reward, reward)

    def accepts(self, coordinates: Coordinates, screen_size: int) -> bool:
        x_is_valid = abs(coordinates[0] - self.previous_coordinates[0]) <= screen_size
        y_is_valid = abs(coordinates[1] - self.previous_coordinates[1]) <= screen_size
        return x_is_valid and y_is_valid

    def revisit_penalty(self, previous_visits: int, settings: ExplorationSettings) -> float:
        grace = max(0, settings.revisit_penalty_grace_steps)
        if settings.staleness_limit <= grace:
            return 0.0
        progress = (self.stale_steps - grace) / (settings.staleness_limit - grace)
        stale_pressure = min(1.0, max(0.0, progress)) ** 2
        visit_pressure = log1p(max(0, previous_visits))
        return settings.step_penalty * visit_pressure * stale_pressure

    def features(self, position: Position, settings: ExplorationSettings) -> list[float]:
        scale = float(settings.screen_size)
        staleness = _unit_interval(self.stale_steps / settings.staleness_limit)
        return [
            _unit_interval((position.x - self.minimum_x) / scale),
            _unit_interval((self.maximum_x - position.x) / scale),
            _unit_interval((position.y - self.minimum_y) / scale),
            _unit_interval((self.maximum_y - position.y) / scale),
            staleness,
        ]


@dataclass(slots=True)
class Exploration:
    coverage: Coverage
    frontier: Frontier
    settings: ExplorationSettings
    invalid_position_steps: int
    discovery_reward_total: float

    @classmethod
    def start(cls, position: Position, settings: ExplorationSettings) -> Self:
        coverage = Coverage.start(position.screen, position.coordinates, settings)
        return cls(coverage, Frontier.start(position.coordinates), settings, 0, 0.0)

    def reward(self, screen: Coordinates, coordinates: Coordinates) -> float:
        if not self.frontier.accepts(coordinates, self.settings.screen_size):
            self.invalid_position_steps += 1
            self.frontier.stale_steps += 1
            return -self.frontier.revisit_penalty(1, self.settings)
        self.frontier.previous_coordinates = coordinates
        discovery = self.coverage.discover(screen, coordinates)
        reward = self.frontier.expand(coordinates, self.settings)
        if discovery.found:
            self.frontier.stale_steps = 0
            reward = max(0.0, reward)
        discovery_reward = self._discovery_reward(discovery)
        self.discovery_reward_total += discovery_reward
        reward += discovery_reward
        if not discovery.position:
            reward -= self.frontier.revisit_penalty(discovery.previous_visits, self.settings)
        return reward

    def completion_reward(self) -> float:
        return max(self.settings.area_reward, self.discovery_reward_total)

    def _discovery_reward(self, discovery: Discovery) -> float:
        reward = self.settings.area_reward if discovery.area else 0.0
        reward += self.settings.region_reward if discovery.region else 0.0
        reward += self.settings.position_reward if discovery.position else 0.0
        return reward


class Explorer(TargetState[T], ABC):
    description: str = ""
    target_missing_penalty: float = 0.0
    exploration: Exploration

    def _on_reset(self) -> None:
        position = self._actor_position(self.ram)
        settings = ExplorationSettings.create(position.screen_size, self.step_penalty)
        self.exploration = Exploration.start(position, settings)
        super()._on_reset()

    def _target_reward(self, distance: float | None) -> float:
        position = self._actor_position(self.ram)
        steps_to_target = self.target_missing_steps + 1
        reward = super()._target_reward(distance)
        reward += self.exploration.reward(position.screen, position.coordinates)
        if distance is None:
            return reward
        self.remember_detected_target()
        completion_reward = self.exploration.completion_reward()
        return reward + completion_reward + self._target_speed_reward(completion_reward, steps_to_target)

    def _target_speed_reward(self, completion_reward: float, steps_to_target: int) -> float:
        return completion_reward / sqrt(max(1, steps_to_target))

    def auxiliary_features(self, ram: T | None) -> list[float]:
        features = super().auxiliary_features(ram)
        if ram is None or not hasattr(self, "exploration"):
            return features + [0.0] * 5
        position = self._actor_position(ram)
        return features + self.exploration.frontier.features(position, self.exploration.settings)

    def _truncated(self) -> bool:
        target_seen = self.target_detector.seen
        stale_steps = self.exploration.frontier.stale_steps
        return stale_steps >= self.exploration.settings.staleness_limit and not target_seen

    def _next(self) -> type[State[T]] | None:
        return self._target_state() if self.target_detector.seen else None

    def _won(self) -> bool:
        return False

    @abstractmethod
    def _target_state(self) -> type[State[T]]:
        pass


def _unit_interval(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
