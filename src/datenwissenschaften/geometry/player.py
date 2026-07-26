import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PlayerGeometry:
    world_x: float
    world_y: float
    screen_x: float
    screen_y: float
    width: int
    height: int
    direction: int

    def __post_init__(self) -> None:
        coordinates = (self.world_x, self.world_y, self.screen_x, self.screen_y)
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError("Player coordinates must be finite")
        if self.width < 1 or self.height < 1:
            raise ValueError("Player dimensions must be positive")
        if self.direction not in {-1, 1}:
            raise ValueError("Player direction must be -1 or 1")

    @property
    def center_x(self) -> float:
        return self.world_x + self.width / 2

    @property
    def bottom(self) -> float:
        return self.world_y + self.height
