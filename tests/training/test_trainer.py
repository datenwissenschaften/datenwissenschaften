from types import SimpleNamespace

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

from datenwissenschaften.training.trainer import _continue_exploration_schedule


def test_continues_exploration_from_current_rate() -> None:
    model = SimpleNamespace(
        num_timesteps=10_000,
        exploration_rate=0.05,
        exploration_final_eps=0.05,
        exploration_schedule=None,
    )

    _continue_exploration_schedule(model)

    assert model.exploration_schedule(0.5) == 0.05


def test_exploration_does_not_restart_between_chunks() -> None:
    environment = DummyVecEnv([lambda: gym.make("CartPole-v1")])
    model = DQN(
        "MlpPolicy",
        environment,
        learning_starts=1_000,
        exploration_fraction=1.0,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
    )
    model.learn(total_timesteps=100, reset_num_timesteps=False)

    _continue_exploration_schedule(model)
    model.learn(total_timesteps=10, reset_num_timesteps=False)

    assert model.exploration_rate == 0.05
