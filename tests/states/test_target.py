from pathlib import Path
from types import SimpleNamespace

import numpy as np

from datenwissenschaften.states.explorer import Explorer
from datenwissenschaften.states.target import TargetState


def target_state() -> TargetState:
    state = TargetState.__new__(TargetState)
    state.target_detector = None
    state.target_memory = SimpleNamespace(coordinates=(100.0, 0.0))
    state.frame = np.zeros((100, 100, 3), dtype=np.uint8)
    state.ram = SimpleNamespace(screen_x=0, screen_y=0, player_x=0, player_y=0)
    state._on_reset()
    return state


def test_target_rewards_progress_instead_of_proximity() -> None:
    state = target_state()
    state.ram.player_x = 10
    assert state._reward() > 0.0
    assert state._reward() == 0.0
    state.ram.player_x = 0
    assert state._reward() < 0.0


def test_explorer_rewards_each_location_once(tmp_path: Path) -> None:
    state = Explorer.__new__(Explorer)
    state.model_dir = tmp_path
    state.target_detector = None
    state.target_memory = SimpleNamespace(coordinates=(100.0, 0.0))
    state.frame = np.zeros((100, 100, 3), dtype=np.uint8)
    state.ram = SimpleNamespace(screen_x=0, screen_y=0, player_x=0, player_y=0)
    state._on_reset()
    assert state._reward() == 1.0
    assert state._reward() == 0.0
