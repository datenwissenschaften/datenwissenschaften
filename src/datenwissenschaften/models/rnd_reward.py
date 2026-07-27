from typing import Any

import numpy as np
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper

from datenwissenschaften.models.rnd import RandomNetworkDistillation


class RNDRewardWrapper(VecEnvWrapper):
    def __init__(self, environment: VecEnv, rnd: RandomNetworkDistillation) -> None:
        super().__init__(environment)
        self.rnd = rnd

    def reset(self) -> Any:
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
        for index in range(len(dones)):
            infos[index]["extrinsic_reward"] = float(rewards[index])
            infos[index]["intrinsic_reward"] = float(intrinsic[index])
        return observations, rewards + coefficient * intrinsic, dones, infos
