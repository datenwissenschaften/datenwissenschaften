import time

from stable_baselines3.common.callbacks import BaseCallback

from datenwissenschaften.ui.telemetry import publish_episode


class EpisodeTelemetryCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.enabled = False
        self.started_at: list[float] = []
        self.fitness: list[float] = []
        self.steps: list[int] = []
        self.training_states: list[str | None] = []

    def _on_training_start(self) -> None:
        self.enabled = hasattr(self.model, "rollout_buffer")
        count = self.training_env.num_envs if self.enabled else 0
        now = time.monotonic()
        self.started_at = [now] * count
        self.fitness = [0.0] * count
        self.steps = [0] * count
        self.training_states = self._state_names(count)

    def _on_step(self) -> bool:
        if not self.enabled:
            return True

        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")
        if rewards is None or dones is None or infos is None:
            return True

        for index, (reward, done, info) in enumerate(zip(rewards, dones, infos, strict=True)):
            self.fitness[index] += float(info.get("extrinsic_reward", reward))
            self.steps[index] += 1
            if not bool(done):
                continue

            monitor_episode = info.get("episode", {})
            fitness = float(monitor_episode.get("r", self.fitness[index]))
            total_steps = int(monitor_episode.get("l", self.steps[index]))
            publish_episode(
                env=index,
                training_state=self.training_states[index] or info.get("state"),
                fitness=fitness,
                training_steps=total_steps,
                total_steps=total_steps,
                duration_seconds=time.monotonic() - self.started_at[index],
                won=None if info.get("won") is None else bool(info.get("won")),
                final_state=info.get("state"),
            )
            self.started_at[index] = time.monotonic()
            self.fitness[index] = 0.0
            self.steps[index] = 0

        if any(bool(done) for done in dones):
            current_states = self._state_names(len(self.training_states))
            for index, done in enumerate(dones):
                if bool(done):
                    self.training_states[index] = current_states[index]

        return True

    def _state_names(self, count: int) -> list[str | None]:
        names = None
        for method_name in ("episode_start_state", "state_name"):
            try:
                names = self.training_env.env_method(method_name)
                break
            except (AttributeError, TypeError, ValueError):
                continue
        if names is None:
            return [None] * count
        return [str(name) if name else None for name in names[:count]] + [None] * max(0, count - len(names))
