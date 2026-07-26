from math import hypot
from pathlib import Path
from typing import TypeVar

from datenwissenschaften.ram.model import RamInfo
from datenwissenschaften.states.image_detector import ImageDetector
from datenwissenschaften.states.target_memory import TargetMemory

T = TypeVar("T", bound=RamInfo)


class TargetState(ImageDetector[T]):
    def __init__(self, model_dir: Path) -> None:
        super().__init__(model_dir)
        self.target_memory = TargetMemory(model_dir / "targets" / f"{type(self).__name__}.json")

    def _reward(self) -> float:
        reward = super()._reward()
        height, width = self.frame.shape[:2]
        actor_x = float(self.ram.screen_x * width + self.ram.position_x)
        actor_y = float(self.ram.screen_y * height + self.ram.position_y)
        if self.target_detector is None or self.target_detector.position is None:
            if self.target_memory.coordinates is None:
                return reward
            target_x, target_y = self.target_memory.coordinates
            distance = hypot(target_x - actor_x, target_y - actor_y)
            return reward + 1.0 / (1.0 + distance / hypot(width, height))
        target_x = float(self.ram.screen_x * width + self.target_detector.position[0])
        target_y = float(self.ram.screen_y * height + self.target_detector.position[1])
        self.target_memory.remember((target_x, target_y))
        distance = hypot(target_x - actor_x, target_y - actor_y)
        return reward + max(0.0, 1.0 - distance / hypot(width, height))
