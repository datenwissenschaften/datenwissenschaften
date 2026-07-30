from math import hypot
from pathlib import Path
from typing import TypeVar

from datenwissenschaften.ram.model import RamInfo
from datenwissenschaften.states.state import State
from datenwissenschaften.states.target_memory import TargetMemory

T = TypeVar("T", bound=RamInfo)
PROGRESS_REWARD_SCALE = 10.0


class TargetState(State[T]):
    def __init__(self, model_dir: Path) -> None:
        super().__init__(model_dir)
        self.target_memory = TargetMemory(model_dir / "targets" / f"{type(self).__name__}.json")
        self.previous_target_distance: float | None = None

    def _on_reset(self) -> None:
        super()._on_reset()
        self.previous_target_distance = self._target_distance()

    def _automatic_reward(self) -> float:
        distance = self._target_distance()
        if distance is None:
            self.previous_target_distance = None
            return 0.0
        previous = self.previous_target_distance
        self.previous_target_distance = distance
        progress = 0.0 if previous is None else previous - distance
        return progress * PROGRESS_REWARD_SCALE

    def _target_distance(self) -> float | None:
        height, width = self.frame.shape[:2]
        actor_x = float(self.ram.screen_x * width + self.ram.player_x)
        actor_y = float(self.ram.screen_y * height + self.ram.player_y)
        if self.target_detector.position is None:
            if self.target_memory.coordinates is None:
                return None
            target_x, target_y = self.target_memory.coordinates
        else:
            target_x = float(self.ram.screen_x * width + self.target_detector.position[0])
            target_y = float(self.ram.screen_y * height + self.target_detector.position[1])
            self.target_memory.remember((target_x, target_y))
        return hypot(target_x - actor_x, target_y - actor_y) / hypot(width, height)
