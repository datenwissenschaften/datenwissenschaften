from pathlib import Path

import gymnasium as gym
import pytest
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from datenwissenschaften.checkpoints.model import atomic_save, atomic_save_replay_buffer
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
    assert model.gradient_steps == -1
    assert sum(parameter.numel() for parameter in model.policy.parameters()) < 100_000


def test_restores_replay_buffer_with_agent(tmp_path: Path) -> None:
    environment = VecFrameStack(DummyVecEnv([cartpole]), n_stack=4)
    path = tmp_path / "model"
    model = load_agent(environment, path)
    model.learn(total_timesteps=32)
    atomic_save_replay_buffer(model, path.with_suffix(".replay.pkl"))
    atomic_save(model, path)

    restored = load_agent(environment, path)

    assert restored.replay_buffer.size() == model.replay_buffer.size()


def test_rejects_checkpoint_without_replay_buffer(tmp_path: Path) -> None:
    environment = VecFrameStack(DummyVecEnv([cartpole]), n_stack=4)
    path = tmp_path / "model"
    atomic_save(load_agent(environment, path), path)

    with pytest.raises(RuntimeError, match="Replay buffer not found"):
        load_agent(environment, path)
