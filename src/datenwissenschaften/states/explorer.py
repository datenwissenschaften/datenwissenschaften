from abc import ABC, abstractmethod
from typing import TypeVar

import numpy as np

from datenwissenschaften.ram import RamInfo
from datenwissenschaften.states.state import State
from datenwissenschaften.states.target import TargetState
from datenwissenschaften.vision.enemy_learner import EnemyLearner, EnemyObservation

T = TypeVar("T", bound=RamInfo)


class Explorer(TargetState[T], ABC):
    description = ""

    area_discovery_reward = 100.0
    position_discovery_reward = 1.0
    target_found_reward = 10000.0
    enemy_discovery_reward = 25.0
    enemy_hit_penalty = 100.0
    enemy_danger_distance = 48.0
    enemy_proximity_penalty_scale = 0.1

    visited_areas: set[tuple[int, int]]
    visited_positions: set[tuple[int, int]]

    def __init__(self) -> None:
        self.enemy_learner = EnemyLearner(self.__class__.__name__)
        self.enemy_features = [0.0, 0.0, 0.0, 0.0]
        self.enemy_seen = False
        self.enemy_distance = 0.0
        self.enemy_hit = False
        super().__init__()

    def _on_reset(self) -> None:
        position = self._actor_position(self.ram)
        self.visited_areas = {position.screen}
        self.visited_positions = {position.coordinates}
        self.enemy_learner.reset()
        self.enemy_features = [0.0, 0.0, 0.0, 0.0]
        self.enemy_seen = False
        self.enemy_distance = 0.0
        self.enemy_hit = False
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
        hit = bool(self._hit())
        enemies = self.enemy_learner.observe(self.frame, position.viewport, hit)
        reward += self._enemy_reward(enemies, hit)
        self._update_enemy_features(enemies, hit)
        return reward

    def auxiliary_features(self, ram: T | None = None) -> list[float]:
        return super().auxiliary_features(ram) + self.enemy_features

    def _hit(self) -> bool:
        """Return a RAM-derived collision signal in the game-specific Explorer."""
        return False

    def _enemy_reward(self, observation: EnemyObservation, hit: bool) -> float:
        """Override to customize learned-enemy rewards and penalties."""
        reward = len(observation.learned_enemy_ids) * self.enemy_discovery_reward
        if hit:
            reward -= self.enemy_hit_penalty
        if observation.detections:
            actor_x, actor_y = self._actor_frame_position()
            nearest = min(
                np.hypot(detection.center[0] - actor_x, detection.center[1] - actor_y)
                for detection in observation.detections
            )
            reward -= max(0.0, self.enemy_danger_distance - nearest) * self.enemy_proximity_penalty_scale
        return reward

    def _update_enemy_features(self, observation: EnemyObservation, hit: bool) -> None:
        frame_height, frame_width = self.frame.shape[:2]
        if observation.detections:
            actor_x, actor_y = self._actor_frame_position()
            nearest = min(
                observation.detections,
                key=lambda detection: np.hypot(
                    detection.center[0] - actor_x,
                    detection.center[1] - actor_y,
                ),
            )
            dx = (nearest.center[0] - actor_x) / max(1, frame_width)
            dy = (nearest.center[1] - actor_y) / max(1, frame_height)
            confidence = nearest.score
            self.enemy_distance = float(np.hypot(nearest.center[0] - actor_x, nearest.center[1] - actor_y))
        else:
            dx = dy = confidence = 0.0
            self.enemy_distance = 0.0
        self.enemy_seen = bool(observation.detections)
        self.enemy_hit = hit
        self.enemy_features = [
            float(np.clip(dx, -1.0, 1.0)),
            float(np.clip(dy, -1.0, 1.0)),
            float(np.clip(confidence, 0.0, 1.0)),
            float(hit),
        ]

    def _actor_frame_position(self) -> tuple[float, float]:
        actor = self._actor_position(self.ram).viewport
        frame_height, frame_width = self.frame.shape[:2]
        return (
            actor.position_x / max(1, actor.screen_size) * frame_width,
            actor.position_y / max(1, actor.screen_size) * frame_height,
        )

    def _next(self) -> type[State[T]] | None:
        return self._target_state() if self.target_detector.seen else None

    def _won(self) -> bool:
        return False

    @abstractmethod
    def _target_state(self) -> type[State[T]]:
        pass
