from pathlib import Path

import gymnasium as gym
import pytest
import torch
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from datenwissenschaften.checkpoints.model import atomic_save
from datenwissenschaften.models.agent import (
    CHECKPOINT_VERSION,
    CPU_INTEROP_THREADS,
    CPU_THREADS,
    HIDDEN_SIZE,
    ROLLOUT_STEPS,
    load_agent,
)


def cartpole() -> gym.Env:
    return gym.make("CartPole-v1")


def state_cartpole() -> gym.Env:
    return gym.wrappers.TransformAction(
        cartpole(),
        lambda action: int(action[0]),
        gym.spaces.MultiBinary(1),
    )


def test_creates_compact_cpu_state_agent(tmp_path: Path) -> None:
    environment = DummyVecEnv([state_cartpole])
    model = load_agent(environment, tmp_path / "model")

    assert isinstance(model, A2C)
    assert model.device == torch.device("cpu")
    assert model.checkpoint_version == CHECKPOINT_VERSION
    assert torch.get_num_threads() == CPU_THREADS
    assert torch.get_num_interop_threads() == CPU_INTEROP_THREADS
    assert model.n_steps == ROLLOUT_STEPS
    assert model.gamma == 0.995
    assert model.ent_coef == 0.01
    assert model.policy.mlp_extractor.policy_net[0].out_features == HIDDEN_SIZE
    assert model.policy.mlp_extractor.value_net[0].out_features == HIDDEN_SIZE
    assert sum(parameter.numel() for parameter in model.policy.parameters()) < 1_000


def test_restores_agent(tmp_path: Path) -> None:
    environment = DummyVecEnv([state_cartpole])
    path = tmp_path / "model"
    model = load_agent(environment, path)
    model.num_timesteps = 123
    atomic_save(model, path)

    restored = load_agent(environment, path)

    assert restored.num_timesteps == 123
    assert restored.device == torch.device("cpu")


def test_restores_agent_with_different_environment_count(tmp_path: Path) -> None:
    path = tmp_path / "model"
    model = load_agent(DummyVecEnv([state_cartpole, state_cartpole]), path)
    model.num_timesteps = 123
    atomic_save(model, path)

    restored = load_agent(DummyVecEnv([state_cartpole]), path)

    assert restored.num_timesteps == 123
    assert restored.n_envs == 1


def test_rejects_old_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "model"
    atomic_save(PPO("MlpPolicy", DummyVecEnv([cartpole]), device="cpu"), path)

    with pytest.raises(RuntimeError, match="Unsupported checkpoint version"):
        load_agent(DummyVecEnv([state_cartpole]), path)


def test_rejects_different_algorithm(tmp_path: Path) -> None:
    path = tmp_path / "model"
    model = DQN("MlpPolicy", DummyVecEnv([cartpole]), device="cpu")
    model.checkpoint_version = CHECKPOINT_VERSION
    atomic_save(model, path)

    with pytest.raises(RuntimeError, match="Unsupported checkpoint algorithm"):
        load_agent(DummyVecEnv([state_cartpole]), path)
