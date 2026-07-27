from typing import Any

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from datenwissenschaften.models.rnd_config import RNDConfig
from datenwissenschaften.models.rnd_network import RNDNetwork


class RandomNetworkDistillation(nn.Module):
    def __init__(self, observation_space: gym.Space, config: RNDConfig, device: torch.device) -> None:
        super().__init__()
        if not isinstance(observation_space, gym.spaces.Dict):
            raise ValueError("RND requires dictionary observations")
        image_space: Any = observation_space.spaces["image"]
        if not isinstance(image_space, gym.spaces.Box) or len(image_space.shape) != 3:
            raise ValueError("RND requires a three-dimensional 'image' observation")
        if image_space.dtype != np.uint8:
            raise ValueError("RND requires uint8 image observations")
        channels = int(image_space.shape[0])
        self.config = config
        self.target = RNDNetwork(channels, config.output_size)
        self.predictor = RNDNetwork(channels, config.output_size)
        self.target.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=config.learning_rate)
        self.register_buffer("reward_mean", torch.zeros(()))
        self.register_buffer("reward_variance", torch.ones(()))
        self.register_buffer("observations_seen", torch.zeros((), dtype=torch.long))
        self.to(device)

    @property
    def coefficient(self) -> float:
        progress = min(int(self.observations_seen.item()) / self.config.anneal_steps, 1.0)
        base = self.config.initial_coefficient + progress * (
            self.config.final_coefficient - self.config.initial_coefficient
        )
        return base

    def intrinsic_rewards(self, images: np.ndarray) -> np.ndarray:
        inputs = torch.as_tensor(images, dtype=torch.float32, device=self.reward_mean.device) / 255.0
        with torch.no_grad():
            targets = self.target(inputs)
        predictions = self.predictor(inputs)
        errors = torch.mean(torch.square(predictions - targets), dim=1)
        self._train_predictor(errors)
        detached = errors.detach()
        self._update_statistics(detached)
        normalized = (detached - self.reward_mean) / torch.sqrt(self.reward_variance + 1e-8)
        rewards = normalized.clamp(0.0, self.config.reward_clip)
        self.observations_seen.add_(len(inputs))
        return rewards.cpu().numpy().astype(np.float32)

    def _train_predictor(self, errors: torch.Tensor) -> None:
        mask = torch.rand_like(errors) < self.config.update_proportion
        if not torch.any(mask):
            mask[torch.randint(len(mask), (), device=mask.device)] = True
        self.optimizer.zero_grad()
        errors[mask].mean().backward()
        nn.utils.clip_grad_norm_(self.predictor.parameters(), 1.0)
        self.optimizer.step()

    @torch.no_grad()
    def _update_statistics(self, rewards: torch.Tensor) -> None:
        momentum = 0.99
        batch_mean = rewards.mean()
        batch_variance = rewards.var(unbiased=False)
        self.reward_mean.mul_(momentum).add_(batch_mean * (1.0 - momentum))
        self.reward_variance.mul_(momentum).add_(batch_variance * (1.0 - momentum))
