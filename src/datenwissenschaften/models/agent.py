from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from loguru import logger
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.policies import RecurrentMultiInputActorCriticPolicy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.save_util import load_from_zip_file
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper
from torch import nn

from datenwissenschaften.rewards.normalizer import REWARD_DISCOUNT_FACTOR


class _RNDNetwork(nn.Module):
    def __init__(self, shape: tuple[int, int, int], output_size: int) -> None:
        super().__init__()
        channels, height, width = shape
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            feature_count = self.encoder(torch.zeros(1, channels, height, width)).shape[1]
        self.head = nn.Sequential(nn.Linear(feature_count, 256), nn.ReLU(), nn.Linear(256, output_size))

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(observation))


class _RNDRewardWrapper(VecEnvWrapper):
    def __init__(self, venv: VecEnv) -> None:
        super().__init__(venv)
        self.rnd: _RND | None = None
        self.enabled = False

    def reset(self) -> Any:
        if self.rnd is not None:
            self.rnd.reset_returns()
        return self.venv.reset()

    def step_wait(self) -> tuple[Any, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        observations, rewards, dones, infos = self.venv.step_wait()
        if not self.enabled or self.rnd is None:
            return observations, rewards, dones, infos
        terminal_observations = [info.get("terminal_observation") for info in infos]
        visual = np.array(observations["scene"], copy=True)
        for index, terminal in enumerate(terminal_observations):
            if terminal is not None:
                visual[index] = terminal["scene"]
        intrinsic = self.rnd.reward(visual, dones)
        combined = rewards.astype(np.float32) + self.rnd.coefficient * intrinsic
        for index, info in enumerate(infos):
            info["extrinsic_reward"] = float(rewards[index])
            info["intrinsic_reward"] = float(intrinsic[index])
            info["intrinsic_reward_coefficient"] = self.rnd.coefficient
        return observations, combined, dones, infos


class _RND(nn.Module):
    def __init__(self, observation_space: gym.spaces.Dict, device: torch.device) -> None:
        super().__init__()
        visual = observation_space.spaces.get("scene")
        if not isinstance(visual, gym.spaces.Box) or len(visual.shape) != 3:
            raise ValueError("RND requires a scene image observation")
        shape = tuple(int(value) for value in visual.shape)
        self.target = _RNDNetwork(shape, 128).to(device)
        self.predictor = _RNDNetwork(shape, 128).to(device)
        self.target.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=1e-4)
        self.device = device
        self.coefficient = 0.5
        self._returns: torch.Tensor | None = None

    def reward(self, observations: np.ndarray, dones: np.ndarray) -> np.ndarray:
        inputs = torch.as_tensor(observations, device=self.device, dtype=torch.float32) / 255.0
        with torch.no_grad():
            target = self.target(inputs)
        predicted = self.predictor(inputs)
        errors = torch.mean((predicted - target.detach()) ** 2, dim=1)
        loss = errors[torch.rand_like(errors) < 0.25]
        if not len(loss):
            loss = errors[:1]
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()
        values = errors.detach()
        if self._returns is None or len(self._returns) != len(values):
            self._returns = torch.zeros_like(values)
        self._returns.mul_(0.99).add_(values)
        normalized = self._returns / torch.sqrt(self._returns.var(unbiased=False) + 1e-8)
        self._returns[torch.as_tensor(dones, device=self.device, dtype=torch.bool)] = 0
        return normalized.clamp(0, 1).cpu().numpy().astype(np.float32)

    def reset_returns(self) -> None:
        self._returns = None


class AdaptiveRecurrentPPO(RecurrentPPO):
    display_name = "Adaptive Recurrent PPO + RND"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.pop("_init_setup_model", None)
        super().__init__(*args, _init_setup_model=False, **kwargs)
        self.env = _RNDRewardWrapper(self.env)
        self.rnd: _RND | None = None
        self._setup_model()
        self.best_fitness: float | None = None
        self.episodes_without_progress = 0
        self.episodes_without_win = 0
        self.exploration_multiplier = 1.0

    def _setup_model(self) -> None:
        super()._setup_model()
        if not isinstance(self.observation_space, gym.spaces.Dict):
            raise ValueError("Adaptive recurrent PPO requires dictionary observations")
        self.rnd = _RND(self.observation_space, self.device)
        self._attach_rnd()

    def set_env(self, env: VecEnv, force_reset: bool = True) -> None:
        super().set_env(_RNDRewardWrapper(env), force_reset=force_reset)
        self._attach_rnd()

    def learn(self, total_timesteps: int, **kwargs: Any) -> Any:
        self.env.enabled = True
        try:
            return super().learn(total_timesteps, **kwargs)
        finally:
            self.env.enabled = False

    def _attach_rnd(self) -> None:
        self.env.rnd = self.rnd

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + ["rnd"]

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts, variables = super()._get_torch_save_params()
        return state_dicts + ["rnd", "rnd.optimizer"], variables

    def record_episode_outcome(self, fitness: float, won: bool) -> None:
        improved = self.best_fitness is None or fitness > self.best_fitness
        self.best_fitness = max(fitness, self.best_fitness or fitness)
        self.episodes_without_progress = 0 if improved else self.episodes_without_progress + 1
        self.episodes_without_win = 0 if won else self.episodes_without_win + 1
        stale = self.episodes_without_progress >= 16
        no_wins = self.episodes_without_win >= 32
        target = 2.0 if stale and no_wins else 1.5 if stale or no_wins else 1.0
        self.exploration_multiplier = min(3.0, self.exploration_multiplier * 0.9 + target * 0.1)
        self.ent_coef = 0.01 * self.exploration_multiplier
        if self.rnd is not None:
            self.rnd.coefficient = 0.5 * self.exploration_multiplier
        for group in self.policy.optimizer.param_groups:
            group["lr"] = 0.00025 / self.exploration_multiplier


def load_agent(environment: Any, path: Path) -> AdaptiveRecurrentPPO:
    checkpoint = path.with_suffix(".zip")
    if checkpoint.is_file():
        _validate_checkpoint(checkpoint)
        logger.info(f"Loading agent from {checkpoint}")
        model = AdaptiveRecurrentPPO.load(checkpoint, env=environment, device="auto")
        model.set_logger(configure(folder=None, format_strings=[]))
        return model
    logger.info("Creating recurrent visual-state PPO agent")
    model = AdaptiveRecurrentPPO(
        "MultiInputLstmPolicy",
        environment,
        device="auto",
        learning_rate=0.00025,
        n_steps=256,
        batch_size=256,
        n_epochs=4,
        gamma=REWARD_DISCOUNT_FACTOR,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.03,
        policy_kwargs={
            "lstm_hidden_size": 128,
            "n_lstm_layers": 1,
            "shared_lstm": True,
            "enable_critic_lstm": False,
            "net_arch": {"pi": [64], "vf": [64]},
            "features_extractor_kwargs": {"cnn_output_dim": 128},
        },
    )
    model.set_logger(configure(folder=None, format_strings=[]))
    return model


def _validate_checkpoint(checkpoint: Path) -> None:
    data, _, _ = load_from_zip_file(checkpoint, device="cpu")
    if "policy_class" not in data:
        raise RuntimeError(f"Invalid checkpoint: {checkpoint}")
    policy_class = data["policy_class"]
    if not isinstance(policy_class, type) or not issubclass(policy_class, RecurrentMultiInputActorCriticPolicy):
        raise RuntimeError(f"Unsupported checkpoint algorithm: {checkpoint}")
