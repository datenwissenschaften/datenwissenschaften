from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

from datenwissenschaften.models.agent import load_agent
from datenwissenschaften.training.model_environment import ModelEnvironment, build_model_environments
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


def test_builds_compact_action_space_for_each_state() -> None:
    source = gym.make("CartPole-v1")
    environments = build_model_environments(
        source.observation_space,
        {
            "Explore": (0, 1, 2, 3),
            "Hold": (0,),
        },
    )

    assert environments["Explore"].action_space.n == 4
    assert environments["Hold"].action_space.n == 1

    for environment in environments.values():
        environment.close()
    source.close()


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
    actions, model_actions = _actions(
        {"First": first, "Second": second},
        ("First", "Second"),
        np.zeros((2, 4), dtype=np.float32),
        {"First": (3, 4), "Second": (5, 6, 7)},
    )

    assert np.array_equal(actions, np.asarray([4, 7]))
    assert np.array_equal(model_actions, np.asarray([1, 2]))
    first.predict.assert_called_once()
    second.predict.assert_called_once()


def test_exploration_samples_only_enabled_state_actions(monkeypatch) -> None:
    model = Mock()
    model.num_timesteps = 0
    model.exploration_initial_eps = 1.0
    model.exploration_final_eps = 1.0
    model.predict.return_value = np.asarray([0]), None
    monkeypatch.setattr("numpy.random.randint", lambda count: count - 1)

    actions, model_actions = _actions(
        {"Scale": model},
        ("Scale",),
        np.zeros((1, 4), dtype=np.float32),
        {"Scale": (0, 5, 9)},
    )

    assert np.array_equal(actions, np.asarray([9]))
    assert np.array_equal(model_actions, np.asarray([2]))


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
