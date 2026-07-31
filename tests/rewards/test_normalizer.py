from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

from datenwissenschaften.rewards.normalizer import (
    REWARD_CLIP,
    REWARD_DISCOUNT_FACTOR,
    normalize_rewards,
    save_reward_normalizer,
)


def environment() -> gym.Env:
    return gym.make("CartPole-v1")


def test_normalizes_rewards_without_normalizing_observations(tmp_path: Path) -> None:
    normalizer = normalize_rewards(DummyVecEnv([environment]), tmp_path)

    assert normalizer.norm_reward
    assert not normalizer.norm_obs
    assert normalizer.gamma == REWARD_DISCOUNT_FACTOR
    assert normalizer.clip_reward == REWARD_CLIP


def test_restores_reward_statistics(tmp_path: Path) -> None:
    normalizer = normalize_rewards(DummyVecEnv([environment]), tmp_path)
    normalizer.ret_rms.update(np.asarray([1.0, 2.0, 3.0]))
    save_reward_normalizer(normalizer, tmp_path)

    restored = normalize_rewards(DummyVecEnv([environment]), tmp_path)

    assert restored.ret_rms.mean == normalizer.ret_rms.mean
    assert restored.ret_rms.var == normalizer.ret_rms.var
    assert restored.ret_rms.count == normalizer.ret_rms.count
