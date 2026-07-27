import gymnasium as gym
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import VecEnv

from datenwissenschaften.models.rnd import RandomNetworkDistillation
from datenwissenschaften.models.rnd_config import RNDConfig
from datenwissenschaften.models.rnd_reward import RNDRewardWrapper


class AdaptiveRecurrentPPO(RecurrentPPO):
    rnd: RandomNetworkDistillation
    rnd_config: RNDConfig

    def configure_rnd(self, config: RNDConfig) -> None:
        self.rnd_config = config
        self.rnd = RandomNetworkDistillation(self.observation_space, config, self.device)
        self._attach_rnd()

    def _setup_model(self) -> None:
        super()._setup_model()
        if hasattr(self, "rnd_config"):
            self.rnd = RandomNetworkDistillation(self.observation_space, self.rnd_config, self.device)
            self._attach_rnd()

    def set_env(self, env: gym.Env | VecEnv, force_reset: bool) -> None:
        super().set_env(env, force_reset=force_reset)
        if hasattr(self, "rnd"):
            self._attach_rnd()

    def _attach_rnd(self) -> None:
        if self.env is None:
            raise RuntimeError("RND requires an environment")
        if isinstance(self.env, RNDRewardWrapper):
            self.env.rnd = self.rnd
            return
        self.env = RNDRewardWrapper(self.env, self.rnd)

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + ["rnd"]

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts, variables = super()._get_torch_save_params()
        return state_dicts + ["rnd", "rnd.optimizer"], variables

    def train(self) -> None:
        if isinstance(self.action_space, gym.spaces.MultiBinary):
            self.rollout_buffer.actions = self.rollout_buffer.actions.astype("float32", copy=False)
        super().train()
