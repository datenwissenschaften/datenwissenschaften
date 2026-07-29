from pathlib import Path

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from datenwissenschaften.models.agent import load_agent


def cartpole() -> gym.Env:
    return gym.make("CartPole-v1")


def test_creates_compact_dqn_agent(tmp_path: Path) -> None:
    environment = VecFrameStack(DummyVecEnv([cartpole, cartpole]), n_stack=4)
    model = load_agent(environment, tmp_path / "model")
    model.learning_starts = 0
    model.learn(total_timesteps=32)

    assert isinstance(model, DQN)
    assert model.buffer_size == 100_000
    assert sum(parameter.numel() for parameter in model.policy.parameters()) < 100_000
