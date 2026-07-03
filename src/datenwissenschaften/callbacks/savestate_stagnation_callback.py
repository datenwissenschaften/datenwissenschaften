import math

from loguru import logger
from stable_baselines3.common.callbacks import BaseCallback


class SavestateStagnationCallback(BaseCallback):
    def __init__(self, total_timesteps: int):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.enabled = False
        self.patience_rollouts = 0
        self.active_savestate: str | None = None
        self.best_fitness = float("-inf")
        self.stagnant_rollouts = 0
        self.episode_fitness: list[float] = []
        self.episode_started_from_automatic_savestate: list[bool] = []
        self.completed_fitness: list[float] = []

    def _on_training_start(self) -> None:
        self.enabled = hasattr(self.model, "rollout_buffer") and hasattr(self.model, "n_steps")
        if not self.enabled:
            return

        rollout_timesteps = max(1, int(self.model.n_steps) * self.training_env.num_envs)
        total_rollouts = math.ceil(self.total_timesteps / rollout_timesteps)
        self.patience_rollouts = max(5, min(20, math.ceil(total_rollouts * 0.1)))
        self.active_savestate = None
        self.best_fitness = float("-inf")
        self.stagnant_rollouts = 0
        self.episode_fitness = [0.0] * self.training_env.num_envs
        self.episode_started_from_automatic_savestate = [False] * self.training_env.num_envs
        self.completed_fitness.clear()
        logger.info(f"PPO automatic-savestate safeguard enabled with {self.patience_rollouts} rollout patience")

    def _on_step(self) -> bool:
        if not self.enabled:
            return True

        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")
        if rewards is None or dones is None or infos is None:
            return True

        for index, (reward, done, info) in enumerate(zip(rewards, dones, infos, strict=True)):
            self.episode_fitness[index] += float(info.get("extrinsic_reward", reward))
            if info.get("started_from_initial_savestate") is False:
                self.episode_started_from_automatic_savestate[index] = True

            if not bool(done):
                continue

            if self.episode_started_from_automatic_savestate[index]:
                monitor_episode = info.get("episode", {})
                fitness = float(monitor_episode.get("r", self.episode_fitness[index]))
                self.completed_fitness.append(fitness)
            self.episode_fitness[index] = 0.0
            self.episode_started_from_automatic_savestate[index] = False

        return True

    def _on_rollout_end(self) -> bool:
        if not self.enabled:
            return True

        active_savestate = self._active_automatic_savestate()
        if active_savestate != self.active_savestate:
            self.active_savestate = active_savestate
            self.best_fitness = float("-inf")
            self.stagnant_rollouts = 0
            self.completed_fitness.clear()
            return True

        if active_savestate is None or not self.completed_fitness:
            self.completed_fitness.clear()
            return True

        fitness = max(self.completed_fitness)
        self.completed_fitness.clear()
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.stagnant_rollouts = 0
            return True

        self.stagnant_rollouts += 1
        if self.stagnant_rollouts < self.patience_rollouts:
            return True

        deleted = self.training_env.env_method("delete_savestate", active_savestate)
        if any(deleted):
            logger.info(
                f"Deleted stagnant PPO savestate for {active_savestate} "
                f"after {self.stagnant_rollouts} rollouts without improvement"
            )
        self.active_savestate = None
        self.best_fitness = float("-inf")
        self.stagnant_rollouts = 0
        return True

    def _active_automatic_savestate(self) -> str | None:
        try:
            states = self.training_env.env_method("active_savestate_state")
        except (AttributeError, TypeError, ValueError):
            return None
        distinct_states = set(states)
        if len(distinct_states) != 1:
            logger.debug(f"PPO environments disagree on the active automatic savestate: {states}")
            return None
        state = distinct_states.pop()
        return str(state) if state else None
