from typing import Any

import numpy as np
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper

from datenwissenschaften.models.rnd import RandomNetworkDistillation


class RNDRewardWrapper(VecEnvWrapper):
    def __init__(self, environment: VecEnv, rnd: RandomNetworkDistillation) -> None:
        super().__init__(environment)
        self.rnd = rnd
        self.episode_returns = np.zeros(environment.num_envs, dtype=np.float32)

    def reset(self) -> Any:
        self.episode_returns.fill(0.0)
        return self.venv.reset()

    def step_wait(self) -> tuple[Any, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        observations, rewards, dones, infos = self.venv.step_wait()
        if not isinstance(observations, dict):
            raise TypeError("RND requires dictionary observations")
        reward_images = np.array(observations["image"], copy=True)
        for index, done in enumerate(dones):
            terminal = infos[index].get("terminal_observation")
            if done and terminal is not None:
                reward_images[index] = terminal["image"]
        intrinsic = self.rnd.intrinsic_rewards(reward_images)
        coefficient = self.rnd.coefficient
        self.episode_returns += rewards
        for index, done in enumerate(dones):
            infos[index]["extrinsic_reward"] = float(rewards[index])
            infos[index]["intrinsic_reward"] = float(intrinsic[index])
            if done:
                self.rnd.record_episode(float(self.episode_returns[index]))
                self.episode_returns[index] = 0.0
        return observations, rewards + coefficient * intrinsic, dones, infos
