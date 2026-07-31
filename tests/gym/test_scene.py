import numpy as np
import pytest

from datenwissenschaften.gym.scene import SCENE_SIZE, scene


def test_scene_converts_rgb_frame_to_compact_grayscale() -> None:
    frame = np.zeros((240, 256, 3), dtype=np.uint8)
    frame[:, :, 0] = 255

    observation = scene(frame)

    assert observation.shape == (1, SCENE_SIZE, SCENE_SIZE)
    assert observation.dtype == np.uint8
    assert np.all(observation == observation[0, 0, 0])


def test_scene_rejects_invalid_frame_shape() -> None:
    with pytest.raises(ValueError, match="two or three dimensions"):
        scene(np.zeros(1, dtype=np.uint8))
