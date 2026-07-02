from __future__ import annotations

import shutil
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from loguru import logger
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper
from torch import nn

from datenwissenschaften.ui.control import (
    consume_model_reset,
    model_reset_requested,
    perform_model_reset,
)

NES_PPO_DEFAULTS: dict[str, Any] = {
    "learning_rate": 1e-4,
    "n_steps": 512,
    "batch_size": 256,
    "n_epochs": 4,
    "gamma": 0.999,
    "gae_lambda": 0.98,
    "clip_range": 0.1,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "policy_kwargs": {
        "lstm_hidden_size": 512,
        "n_lstm_layers": 1,
        "shared_lstm": False,
        "enable_critic_lstm": True,
    },
}


class _RNDNetwork(nn.Module):
    def __init__(self, observation_shape: tuple[int, int, int], output_size: int) -> None:
        super().__init__()
        channels, height, width = observation_shape
        self.convolution = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            flattened_size = self.convolution(torch.zeros(1, channels, height, width)).shape[1]
        self.projection = nn.Sequential(
            nn.Linear(flattened_size, 512),
            nn.LeakyReLU(),
            nn.Linear(512, output_size),
        )
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.projection(self.convolution(observations))


class _RandomNetworkDistillation(nn.Module):
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        *,
        output_size: int,
        learning_rate: float,
        update_proportion: float,
        intrinsic_gamma: float,
        intrinsic_coefficient: float,
        final_intrinsic_coefficient: float,
        anneal_steps: int,
        reward_clip: float,
        device: torch.device,
    ) -> None:
        super().__init__()
        visual_space = observation_space.spaces.get("visual")
        if not isinstance(visual_space, gym.spaces.Box) or len(visual_space.shape) != 3:
            raise ValueError("RND requires a 'visual' image observation with shape (C, H, W).")
        if not 0 < update_proportion <= 1:
            raise ValueError("rnd_update_proportion must be in (0, 1].")
        if not 0 <= intrinsic_gamma <= 1:
            raise ValueError("rnd_gamma must be in [0, 1].")
        if intrinsic_coefficient < 0 or final_intrinsic_coefficient < 0:
            raise ValueError("RND intrinsic reward coefficients must be non-negative.")
        if anneal_steps < 1:
            raise ValueError("rnd_anneal_steps must be positive.")
        if reward_clip <= 0:
            raise ValueError("rnd_reward_clip must be positive.")

        shape = tuple(int(value) for value in visual_space.shape)
        self.target = _RNDNetwork(shape, output_size)
        self.predictor = _RNDNetwork(shape, output_size)
        self.target.requires_grad_(False)
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=learning_rate)

        self.update_proportion = float(update_proportion)
        self.intrinsic_gamma = float(intrinsic_gamma)
        self.initial_coefficient = float(intrinsic_coefficient)
        self.final_coefficient = float(final_intrinsic_coefficient)
        self.anneal_steps = int(anneal_steps)
        self.reward_clip = float(reward_clip)
        self._observation_scale = 255.0 if visual_space.dtype == np.uint8 else 1.0
        self._returns: torch.Tensor | None = None

        self.register_buffer("reward_mean", torch.zeros((), dtype=torch.float64))
        self.register_buffer("reward_variance", torch.ones((), dtype=torch.float64))
        self.register_buffer("reward_count", torch.tensor(1e-4, dtype=torch.float64))
        self.register_buffer("observations_seen", torch.zeros((), dtype=torch.long))
        self.to(device)

    @property
    def coefficient(self) -> float:
        fraction = min(int(self.observations_seen.item()) / self.anneal_steps, 1.0)
        return self.initial_coefficient + fraction * (self.final_coefficient - self.initial_coefficient)

    def intrinsic_reward(self, observations: dict[str, np.ndarray], dones: np.ndarray) -> tuple[np.ndarray, float]:
        inputs = torch.as_tensor(observations["visual"], device=self.reward_mean.device, dtype=torch.float32)
        inputs = inputs / self._observation_scale
        with torch.no_grad():
            target_features = self.target(inputs)
        predicted_features = self.predictor(inputs)
        raw_reward = torch.mean(torch.square(predicted_features.detach() - target_features), dim=1)

        self._train_predictor(predicted_features, target_features)
        self._update_reward_statistics(raw_reward, dones)
        normalized = raw_reward / torch.sqrt(self.reward_variance.float() + 1e-8)
        normalized = torch.clamp(normalized, min=0.0, max=self.reward_clip)
        self.observations_seen.add_(len(inputs))
        return normalized.cpu().numpy().astype(np.float32), self.coefficient

    def reset_returns(self) -> None:
        self._returns = None

    def _train_predictor(self, predicted: torch.Tensor, target: torch.Tensor) -> None:
        per_sample_loss = torch.mean(torch.square(predicted - target.detach()), dim=1)
        mask = torch.rand_like(per_sample_loss) < self.update_proportion
        if not torch.any(mask):
            mask[torch.randint(len(mask), (), device=mask.device)] = True
        loss = per_sample_loss[mask].mean()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.predictor.parameters(), 1.0)
        self.optimizer.step()

    @torch.no_grad()
    def _update_reward_statistics(self, rewards: torch.Tensor, dones: np.ndarray) -> None:
        if self._returns is None or len(self._returns) != len(rewards):
            self._returns = torch.zeros_like(rewards)
        self._returns.mul_(self.intrinsic_gamma).add_(rewards)
        values = self._returns.double()
        batch_mean = values.mean()
        batch_variance = values.var(unbiased=False)
        batch_count = torch.tensor(float(len(values)), dtype=torch.float64, device=values.device)

        delta = batch_mean - self.reward_mean
        total_count = self.reward_count + batch_count
        new_mean = self.reward_mean + delta * batch_count / total_count
        first_moment = self.reward_variance * self.reward_count
        second_moment = batch_variance * batch_count
        combined = first_moment + second_moment + delta.square() * self.reward_count * batch_count / total_count
        self.reward_mean.copy_(new_mean)
        self.reward_variance.copy_(combined / total_count)
        self.reward_count.copy_(total_count)

        done_tensor = torch.as_tensor(dones, device=values.device, dtype=torch.bool)
        self._returns[done_tensor] = 0


class _RNDRewardWrapper(VecEnvWrapper):
    def __init__(self, venv: VecEnv) -> None:
        super().__init__(venv)
        self.rnd: _RandomNetworkDistillation | None = None
        self.enabled = False

    def reset(self):
        if self.rnd is not None:
            self.rnd.reset_returns()
        return self.venv.reset()

    def step_wait(self):
        observations, rewards, dones, infos = self.venv.step_wait()
        if self.rnd is None or not self.enabled:
            return observations, rewards, dones, infos

        if not isinstance(observations, dict):
            raise TypeError("RecurrentRNDPPO requires visual and RAM dictionary observations.")
        reward_observations = {key: np.array(value, copy=True) for key, value in observations.items()}
        for index, done in enumerate(dones):
            terminal_observation = infos[index].get("terminal_observation")
            if done and terminal_observation is not None:
                for key, value in terminal_observation.items():
                    reward_observations[key][index] = value

        intrinsic_rewards, coefficient = self.rnd.intrinsic_reward(reward_observations, dones)
        combined_rewards = rewards.astype(np.float32, copy=False) + coefficient * intrinsic_rewards
        for index, info in enumerate(infos):
            info["extrinsic_reward"] = float(rewards[index])
            info["intrinsic_reward"] = float(intrinsic_rewards[index])
            info["intrinsic_reward_coefficient"] = coefficient
        return observations, combined_rewards, dones, infos


class _StopOnModelResetCallback(BaseCallback):
    def _on_step(self) -> bool:
        return not model_reset_requested()


class RecurrentRNDPPO(RecurrentPPO):
    supports_ui_restart = True

    def __init__(
        self,
        policy: str | type = "MultiInputLstmPolicy",
        env: VecEnv | str | None = None,
        *,
        rnd_output_size: int = 256,
        rnd_learning_rate: float = 1e-4,
        rnd_update_proportion: float = 0.25,
        rnd_gamma: float = 0.999,
        rnd_intrinsic_coefficient: float = 1.0,
        rnd_final_intrinsic_coefficient: float = 0.05,
        rnd_anneal_steps: int = 10_000_000,
        rnd_reward_clip: float = 5.0,
        _init_setup_model: bool = True,
        **kwargs: Any,
    ) -> None:
        if (
            policy != "MultiInputLstmPolicy"
            and getattr(policy, "__name__", "") != "RecurrentMultiInputActorCriticPolicy"
        ):
            raise ValueError("RecurrentRNDPPO requires MultiInputLstmPolicy.")

        self.rnd_output_size = int(rnd_output_size)
        self.rnd_learning_rate = float(rnd_learning_rate)
        self.rnd_update_proportion = float(rnd_update_proportion)
        self.rnd_gamma = float(rnd_gamma)
        self.rnd_intrinsic_coefficient = float(rnd_intrinsic_coefficient)
        self.rnd_final_intrinsic_coefficient = float(rnd_final_intrinsic_coefficient)
        self.rnd_anneal_steps = int(rnd_anneal_steps)
        self.rnd_reward_clip = float(rnd_reward_clip)
        self.rnd: _RandomNetworkDistillation | None = None

        for name, value in NES_PPO_DEFAULTS.items():
            kwargs.setdefault(name, value.copy() if isinstance(value, dict) else value)
        self._restart_kwargs = {
            name: value.copy() if isinstance(value, dict) else value for name, value in kwargs.items()
        }

        # Let SB3 convert Gym environments and registered environment names to
        # VecEnv before installing the reward wrapper.
        super().__init__(policy, env, _init_setup_model=False, **kwargs)
        if self.env is not None and not isinstance(self.env, _RNDRewardWrapper):
            self.env = _RNDRewardWrapper(self.env)
        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        if not isinstance(self.observation_space, gym.spaces.Dict):
            raise ValueError("RecurrentRNDPPO requires visual and RAM dictionary observations.")
        self.rnd = _RandomNetworkDistillation(
            self.observation_space,
            output_size=self.rnd_output_size,
            learning_rate=self.rnd_learning_rate,
            update_proportion=self.rnd_update_proportion,
            intrinsic_gamma=self.rnd_gamma,
            intrinsic_coefficient=self.rnd_intrinsic_coefficient,
            final_intrinsic_coefficient=self.rnd_final_intrinsic_coefficient,
            anneal_steps=self.rnd_anneal_steps,
            reward_clip=self.rnd_reward_clip,
            device=self.device,
        )
        self._attach_rnd_to_env()

    def set_env(self, env: VecEnv, force_reset: bool = True) -> None:
        if not isinstance(env, _RNDRewardWrapper):
            env = _RNDRewardWrapper(env)
        super().set_env(env, force_reset=force_reset)
        self._attach_rnd_to_env()

    def learn(
        self,
        total_timesteps: int,
        callback=None,
        log_interval: int | None = None,
        tb_log_name: str = "RecurrentPPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        restart_budget = total_timesteps + (self.num_timesteps if not reset_num_timesteps else 0)
        reset_callback = _StopOnModelResetCallback()
        if callback is None:
            active_callbacks = [reset_callback]
        elif isinstance(callback, list):
            active_callbacks = [reset_callback, *callback]
        else:
            active_callbacks = [reset_callback, callback]

        wrapper = self.env if isinstance(self.env, _RNDRewardWrapper) else None
        if wrapper is not None:
            wrapper.enabled = True
        try:
            result = super().learn(
                total_timesteps=total_timesteps,
                callback=active_callbacks,
                log_interval=log_interval,
                tb_log_name=tb_log_name,
                reset_num_timesteps=reset_num_timesteps,
                progress_bar=progress_bar,
            )
        finally:
            if wrapper is not None:
                wrapper.enabled = False

        reset = consume_model_reset()
        if reset is None:
            return result
        if wrapper is None:
            raise RuntimeError("Cannot restart RecurrentRNDPPO without an environment.")

        perform_model_reset(reset)
        restart_kwargs = dict(self._restart_kwargs)
        restart_kwargs["device"] = self.device
        fresh_model = type(self)(
            env=wrapper.venv,
            rnd_output_size=self.rnd_output_size,
            rnd_learning_rate=self.rnd_learning_rate,
            rnd_update_proportion=self.rnd_update_proportion,
            rnd_gamma=self.rnd_gamma,
            rnd_intrinsic_coefficient=self.rnd_intrinsic_coefficient,
            rnd_final_intrinsic_coefficient=self.rnd_final_intrinsic_coefficient,
            rnd_anneal_steps=self.rnd_anneal_steps,
            rnd_reward_clip=self.rnd_reward_clip,
            **restart_kwargs,
        )
        return fresh_model.learn(
            total_timesteps=restart_budget,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=True,
            progress_bar=progress_bar,
        )

    def train(self) -> None:
        if isinstance(self.action_space, gym.spaces.MultiBinary):
            self.rollout_buffer.actions = self.rollout_buffer.actions.astype(np.float32, copy=False)
        super().train()

    def _attach_rnd_to_env(self) -> None:
        if isinstance(self.env, _RNDRewardWrapper):
            self.env.rnd = self.rnd

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + ["rnd"]

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts, variables = super()._get_torch_save_params()
        return state_dicts + ["rnd", "rnd.optimizer"], variables


def build_recurrent_rnd_ppo(env: VecEnv, **kwargs: Any) -> RecurrentRNDPPO:
    kwargs.setdefault("verbose", 0)
    return RecurrentRNDPPO("MultiInputLstmPolicy", env, **kwargs)


class RecurrentRNDModel:
    @staticmethod
    def cleanup_incompatible_artifacts(config) -> None:
        game_dir = config.paths.models_dir / config.training.game
        for name in ("datenwissenschaften", "neat"):
            path = game_dir / name
            if path.is_symlink() or path.is_file():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                shutil.rmtree(path)
            else:
                continue
            logger.info(f"Removed incompatible NEAT model artifacts: {path}")

    @staticmethod
    def load(path: str, **kwargs: Any) -> RecurrentRNDPPO:
        return RecurrentRNDPPO.load(path, **kwargs)

    def __call__(self, env: VecEnv) -> RecurrentRNDPPO:
        return build_recurrent_rnd_ppo(env)
