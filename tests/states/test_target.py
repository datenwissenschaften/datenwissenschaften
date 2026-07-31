from pathlib import Path
from types import SimpleNamespace

import numpy as np

from datenwissenschaften.states.explorer import Explorer
from datenwissenschaften.states.target import TargetState


def target_state() -> TargetState:
    state = TargetState.__new__(TargetState)
    state.target_detector = SimpleNamespace(position=None)
    state.target_memory = SimpleNamespace(coordinates=(100.0, 0.0))
    state.frame = np.zeros((100, 100, 3), dtype=np.uint8)
    state.ram = SimpleNamespace(screen_x=0, screen_y=0, player_x=0, player_y=0)
    state._on_reset()
    return state


def test_target_rewards_progress_and_remaining_near_target() -> None:
    state = target_state()
    state.ram.player_x = 10
    approaching_reward = state._automatic_reward()
    stationary_reward = state._automatic_reward()
    assert approaching_reward > stationary_reward
    state.ram.player_x = 100
    assert state._automatic_reward() > 0.0
    assert state._automatic_reward() == 0.0
    state.ram.player_x = 0
    assert state._automatic_reward() < 0.0


def test_target_features_include_remembered_relative_direction() -> None:
    state = target_state()

    assert np.array_equal(state.target_features(), np.asarray((1.0, 1.0, 0.0), dtype=np.float32))
    state.ram.player_x = 50
    assert np.array_equal(state.target_features(), np.asarray((1.0, 0.5, 0.0), dtype=np.float32))


def test_explorer_rewards_current_tile_once(tmp_path: Path) -> None:
    state = Explorer.__new__(Explorer)
    state.model_dir = tmp_path
    state.target_detector = SimpleNamespace(position=None)
    state.target_memory = SimpleNamespace(coordinates=(100.0, 0.0))
    state.frame = np.zeros((100, 100, 3), dtype=np.uint8)
    state.ram = SimpleNamespace(screen_x=0, screen_y=0, player_x=0, player_y=0)
    state._on_reset()
    first_visit_reward = state._automatic_reward()
    repeated_visit_reward = state._automatic_reward()
    assert np.isclose(first_visit_reward - repeated_visit_reward, 0.1)


def test_explorer_rewards_each_tile_once(tmp_path: Path) -> None:
    state = Explorer.__new__(Explorer)
    state.model_dir = tmp_path
    state.target_detector = SimpleNamespace(position=None)
    state.target_memory = SimpleNamespace(coordinates=None)
    state.frame = np.zeros((100, 100, 3), dtype=np.uint8)
    state.ram = SimpleNamespace(screen_x=0, screen_y=0, player_x=0, player_y=0)
    state._on_reset()

    assert state._automatic_reward() == 0.1
    state.ram.player_x = 7
    assert state._automatic_reward() == 0.0
    state.ram.player_x = 8
    assert state._automatic_reward() == 0.1
