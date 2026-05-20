from stable_baselines3.common.callbacks import BaseCallback


class StopTrainingAtTimestepsCallback(BaseCallback):
    def __init__(self, total_timesteps: int):
        super().__init__()
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        return self.model.num_timesteps < self.total_timesteps
