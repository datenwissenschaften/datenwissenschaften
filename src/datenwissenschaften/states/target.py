from abc import ABC
from pathlib import Path
from typing import ClassVar, TypeVar

from datenwissenschaften.helpers.position import Position
from datenwissenschaften.ram import RamInfo
from datenwissenschaften.settings import load_config
from datenwissenschaften.states.state import State
from datenwissenschaften.states.target_memory import TargetMemory
from datenwissenschaften.vision.template_detector import TemplateDetector

T = TypeVar("T", bound=RamInfo)


class TargetState(State[T], ABC):
    description = ""
    progress = -1

    template_file: ClassVar[str]
    stay_near_distance = 80.0
    target_missing_penalty = 2.0
    step_penalty = 0.05
    proximity_reward_scale = 0.05

    target_missing_steps: int

    def __init__(self) -> None:
        self.target_detector = TemplateDetector(self.template_file)
        super().__init__()
        self.target_memory = TargetMemory.shared(
            self._target_memory_path(),
            origin=(0.0, 0.0),
            scale=float(Position.screen_size),
        )

    def auxiliary_features(self, ram: T | None = None) -> list[float]:
        coordinates = None if ram is None else self._actor_position(ram).coordinates
        return self.target_memory.features(coordinates)

    def remember_detected_target(self) -> bool:
        if not self.target_detector.seen or self.target_detector.position is None:
            return False
        actor_position = self._actor_position(self.ram)
        coordinates = self.target_detector.position.on_screen(actor_position.screen).coordinates
        return self.target_memory.remember(coordinates)

    def _on_reset(self) -> None:
        self.target_missing_steps = 0
        super()._on_reset()
        self._reset_state()

    def _reset_state(self) -> None:
        pass

    def _reward(self) -> float:
        self.target_detector.detect(self.frame)
        distance = self.target_detector.distance(self._actor_position(self.ram).viewport)
        return self._target_reward(distance)

    def _target_reward(self, distance: float | None) -> float:
        reward = -self.step_penalty
        if distance is None:
            self.target_missing_steps += 1
            reward -= self.target_missing_penalty
        else:
            self.target_missing_steps = 0
            reward += max(0.0, self.stay_near_distance - distance) * self.proximity_reward_scale
        return reward + self._additional_target_reward(distance)

    def _additional_target_reward(self, distance: float | None) -> float:
        return 0.0

    def _target_memory_path(self) -> str | Path:
        return load_config().paths.cache_dir / "target_memory" / f"{Path(self.template_file).stem}.json"

    def _actor_position(self, ram: T) -> Position:
        return Position(
            getattr(ram, "position_x", 0),
            getattr(ram, "position_y", 0),
            getattr(ram, "screen_x", 0),
            getattr(ram, "screen_y", 0),
        )
