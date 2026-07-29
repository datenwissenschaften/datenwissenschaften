from types import SimpleNamespace

import numpy as np

from datenwissenschaften.gym.player_motion import PlayerMotion


def test_calculates_velocity_from_global_player_position() -> None:
    motion = PlayerMotion()
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    ram = SimpleNamespace(screen_x=0, screen_y=0, player_x=190, player_y=50)

    assert np.array_equal(motion.reset(ram, frame), np.zeros(2, dtype=np.float32))
    ram.screen_x = 1
    ram.player_x = 10
    ram.player_y = 40

    assert np.allclose(motion.measure(ram, frame), np.asarray((0.1, -0.1), dtype=np.float32))
