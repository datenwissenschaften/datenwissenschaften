from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from sb3_contrib import RecurrentPPO
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

from datenwissenschaften.checkpoints.model import atomic_save
from datenwissenschaften.gym.scene import SCENE_SIZE
from datenwissenschaften.models.agent import load_agent

VISUAL_OBSERVATION_SPACE = gym.spaces.Dict(
    {
        "scene": gym.spaces.Box(0, 255, shape=(1, SCENE_SIZE, SCENE_SIZE), dtype=np.uint8),
        "state": gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32),
    }
)


def cartpole() -> gym.Env:
    return gym.make("CartPole-v1")


def visual_observation(state: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "scene": np.zeros((1, SCENE_SIZE, SCENE_SIZE), dtype=np.uint8),
        "state": state.astype(np.float32),
    }


def visual_cartpole() -> gym.Env:
    return gym.wrappers.TransformObservation(
        cartpole(),
        visual_observation,
        VISUAL_OBSERVATION_SPACE,
    )


def test_creates_compact_recurrent_agent(tmp_path: Path) -> None:
    environment = DummyVecEnv([visual_cartpole, visual_cartpole])
    model = load_agent(environment, tmp_path / "model")

    assert isinstance(model, RecurrentPPO)
    assert model.n_steps == 256
    assert model.batch_size == 256
    assert model.n_epochs == 4
    assert model.gamma == 0.995
    assert model.ent_coef == 0.01
    assert model.policy.lstm_actor.hidden_size == 128
    assert model.policy.shared_lstm
    assert sum(parameter.numel() for parameter in model.policy.parameters()) < 750_000


def test_restores_agent(tmp_path: Path) -> None:
    environment = DummyVecEnv([visual_cartpole])
    path = tmp_path / "model"
    model = load_agent(environment, path)
    model.num_timesteps = 123
    atomic_save(model, path)

    restored = load_agent(environment, path)

    assert restored.num_timesteps == 123


def test_restores_agent_with_different_environment_count(tmp_path: Path) -> None:
    path = tmp_path / "model"
    model = load_agent(DummyVecEnv([visual_cartpole, visual_cartpole]), path)
    model.num_timesteps = 123
    atomic_save(model, path)

    restored = load_agent(DummyVecEnv([visual_cartpole]), path)

    assert restored.num_timesteps == 123
    assert restored.n_envs == 1


def test_rejects_dqn_checkpoint(tmp_path: Path) -> None:
    environment = DummyVecEnv([cartpole])
    path = tmp_path / "model"
    atomic_save(DQN("MlpPolicy", environment), path)

    with pytest.raises(RuntimeError, match="Unsupported checkpoint algorithm"):
        load_agent(environment, path)


def test_rejects_feature_only_recurrent_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "model"
    atomic_save(RecurrentPPO("MlpLstmPolicy", DummyVecEnv([cartpole]), device="cpu"), path)

    with pytest.raises(RuntimeError, match="Unsupported checkpoint algorithm"):
        load_agent(DummyVecEnv([visual_cartpole]), path)
