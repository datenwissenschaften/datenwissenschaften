from math import hypot
from pathlib import Path
from typing import ClassVar, TypeVar

import numpy as np

from datenwissenschaften.ram.model import RamInfo
from datenwissenschaften.states.detection import (
    MATCH_THRESHOLD,
    MINIMUM_TARGET_DISTANCE,
)
from datenwissenschaften.states.multi_template_detector import MultiTemplateDetector
from datenwissenschaften.states.state import State

T = TypeVar("T", bound=RamInfo)
DISTANCE_REWARD_SCALE = 10.0
TARGET_SWITCH_DISTANCE = 8.0


class RamScorerState(State[T]):
    detector_files: ClassVar[tuple[str, ...]] = ()
    target_detector: MultiTemplateDetector | None = None
    previous_value: float | None = None

    def __init__(self, model_dir: Path) -> None:
        super().__init__(model_dir)
        self.target_detector = (
            MultiTemplateDetector(
                self.detector_files,
                MATCH_THRESHOLD,
                MINIMUM_TARGET_DISTANCE,
            )
            if self.detector_files
            else None
        )
        self.previous_target_position: tuple[float, float] | None = None
        self.previous_target_distance: float | None = None

    def _detect(self) -> None:
        if self.target_detector is not None:
            self.target_detector.detect(self.frame)

    def _on_reset(self) -> None:
        super()._on_reset()
        self.previous_value = float(self._scored_value())
        target = self._target_coordinates()
        self.previous_target_position = target
        self.previous_target_distance = None if target is None else self._distance_to(target)

    def _automatic_reward(self) -> float:
        current_value = float(self._scored_value())
        previous = self.previous_value
        self.previous_value = current_value

        if previous is None:
            score_reward = 0.0
        else:
            score_reward = current_value - previous

        target = self._target_coordinates()
        previous_target = self.previous_target_position
        previous_distance = self.previous_target_distance
        distance = None if target is None else self._distance_to(target)
        self.previous_target_position = target
        self.previous_target_distance = distance

        if target is None or previous_target is None or previous_distance is None:
            return score_reward
        if hypot(target[0] - previous_target[0], target[1] - previous_target[1]) >= TARGET_SWITCH_DISTANCE:
            return score_reward
        return score_reward + (previous_distance - distance) * DISTANCE_REWARD_SCALE

    def target_features(self) -> np.ndarray:
        target = self._target_coordinates()
        if target is None:
            return np.zeros(3, dtype=np.float32)
        height, width = self.frame.shape[:2]
        actor_x = float(self.ram.screen_x * width + self.ram.player_x)
        actor_y = float(self.ram.screen_y * height + self.ram.player_y)
        target_x, target_y = target
        return np.asarray(
            (
                1.0,
                np.clip((target_x - actor_x) / width, -1.0, 1.0),
                np.clip((target_y - actor_y) / height, -1.0, 1.0),
            ),
            dtype=np.float32,
        )

    def _distance_to(self, target: tuple[float, float]) -> float:
        height, width = self.frame.shape[:2]
        actor_x = float(self.ram.screen_x * width + self.ram.player_x)
        actor_y = float(self.ram.screen_y * height + self.ram.player_y)
        return hypot(target[0] - actor_x, target[1] - actor_y) / hypot(width, height)

    def _target_coordinates(self) -> tuple[float, float] | None:
        if self.target_detector is None or not self.target_detector.positions:
            return None
        height, width = self.frame.shape[:2]
        actor_x = float(self.ram.screen_x * width + self.ram.player_x)
        actor_y = float(self.ram.screen_y * height + self.ram.player_y)
        candidates = tuple(
            (
                float(self.ram.screen_x * width + position[0]),
                float(self.ram.screen_y * height + position[1]),
            )
            for position in self.target_detector.positions
        )
        return min(candidates, key=lambda position: hypot(position[0] - actor_x, position[1] - actor_y))

    def _scored_value(self) -> float:
        raise NotImplementedError
