from dataclasses import dataclass
from math import hypot


@dataclass(frozen=True)
class Position:
    position_x: int
    position_y: int
    screen_x: int = 0
    screen_y: int = 0
    screen_size: int = 256

    @property
    def x(self) -> int:
        return self.screen_x * self.screen_size + self.position_x

    @property
    def y(self) -> int:
        return self.screen_y * self.screen_size + self.position_y

    @property
    def is_zero(self) -> bool:
        return self.position_x == 0 and self.position_y == 0 and self.screen_x == 0 and self.screen_y == 0

    @staticmethod
    def _is_invalid(position: "Position | None") -> bool:
        return position is None or position.is_zero

    def distance_to(self, other: "Position | None") -> float:
        if self._is_invalid(other):
            return 0.0
        return hypot(other.x - self.x, other.y - self.y)

    def speed_to(self, previous: "Position | None", dt: float = 1.0) -> float:
        if self._is_invalid(previous):
            return 0.0
        return (
            hypot(
                self.x - previous.x,
                self.y - previous.y,
            )
            / dt
        )

    def __sub__(self, other: "Position | None") -> tuple[int, int]:
        if self._is_invalid(other):
            return 0, 0
        return self.x - other.x, self.y - other.y
