from pathlib import Path
from typing import TypeVar

from datenwissenschaften.ram.model import RamInfo
from datenwissenschaften.states.state import State

T = TypeVar("T", bound=RamInfo)


class ImageDetector(State[T]):
    def __init__(self, model_dir: Path) -> None:
        super().__init__(model_dir)
        if self.target_detector is None:
            raise TypeError(f"{type(self).__name__} must define template_file")
