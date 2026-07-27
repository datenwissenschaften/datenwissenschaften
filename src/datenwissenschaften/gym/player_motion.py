import numpy as np

from datenwissenschaften.ram.model import RamInfo


class PlayerMotion:
    def __init__(self) -> None:
        self.previous_position: tuple[float, float] | None = None

    def reset(self, ram: RamInfo, frame: np.ndarray) -> np.ndarray:
        self.previous_position = self._position(ram, frame)
        return np.zeros(2, dtype=np.float32)

    def measure(self, ram: RamInfo, frame: np.ndarray) -> np.ndarray:
        current = self._position(ram, frame)
        previous = self.previous_position
        self.previous_position = current
        if previous is None:
            return np.zeros(2, dtype=np.float32)
        height, width = frame.shape[:2]
        velocity_x = np.clip((current[0] - previous[0]) / width, -1.0, 1.0)
        velocity_y = np.clip((current[1] - previous[1]) / height, -1.0, 1.0)
        return np.asarray((velocity_x, velocity_y), dtype=np.float32)

    @staticmethod
    def _position(ram: RamInfo, frame: np.ndarray) -> tuple[float, float]:
        height, width = frame.shape[:2]
        player_x = float(getattr(ram, "player_x"))
        player_y = float(getattr(ram, "player_y"))
        screen_x = float(getattr(ram, "screen_x"))
        screen_y = float(getattr(ram, "screen_y"))
        return screen_x * width + player_x, screen_y * height + player_y
