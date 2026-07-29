from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

from datenwissenschaften.models.agent import load_agent
from datenwissenschaften.training.model_environment import ModelEnvironment
from datenwissenschaften.training.trainer import _actions, _exploration_rate, _learn


def test_exploration_decays_per_state_model() -> None:
    model = SimpleNamespace(
        num_timesteps=50_000,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        exploration_rate=1.0,
    )

    rate = _exploration_rate(model)

    assert rate == 0.525
    assert model.exploration_rate == rate


def test_routes_each_environment_to_its_state_model() -> None:
    first = Mock()
    first.num_timesteps = 100_000
    first.exploration_initial_eps = 1.0
    first.exploration_final_eps = 0.0
    first.predict.return_value = np.asarray([1]), None
    second = Mock()
    second.num_timesteps = 100_000
    second.exploration_initial_eps = 1.0
    second.exploration_final_eps = 0.0
    second.predict.return_value = np.asarray([2]), None
    action_space = Mock()

    actions = _actions(
        {"First": first, "Second": second},
        ("First", "Second"),
        np.zeros((2, 4), dtype=np.float32),
        action_space,
    )

    assert np.array_equal(actions, np.asarray([1, 2]))
    first.predict.assert_called_once()
    second.predict.assert_called_once()


def test_transition_is_terminal_for_outgoing_state_model() -> None:
    replay_buffer = Mock()
    model = Mock()
    model.replay_buffer = replay_buffer
    model.num_timesteps = 0
    model.learning_starts = 5_000
    model.target_update_interval = 10_000

    _learn(
        {"First": model},
        ("First",),
        np.zeros((1, 4), dtype=np.float32),
        np.asarray([1]),
        np.ones((1, 4), dtype=np.float32),
        np.asarray([5.0]),
        np.asarray([False]),
        [{"state": "Second"}],
    )

    stored = replay_buffer.add.call_args.args
    assert stored[4].item()
    assert model.num_timesteps == 1
    model.train.assert_not_called()


def test_trains_state_model_outside_sb3_learn_loop(tmp_path: Path) -> None:
    source = gym.make("CartPole-v1")
    environment = DummyVecEnv([lambda: ModelEnvironment(source.observation_space, source.action_space)])
    model = load_agent(environment, tmp_path / "model")
    model.learning_starts = 0

    _learn(
        {"First": model},
        ("First",),
        np.zeros((1, 4), dtype=np.float32),
        np.asarray([1]),
        np.ones((1, 4), dtype=np.float32),
        np.asarray([1.0]),
        np.asarray([False]),
        [{"state": "First"}],
    )

    assert model.num_timesteps == 1
    assert model.replay_buffer.size() == 1
    assert model._n_updates == 1
