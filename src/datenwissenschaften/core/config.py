from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    game: str
    num_envs: int
    n_stack: int
    population_size: int
    timestep_batch: int = int(1e7)

    def __post_init__(self) -> None:
        if not self.game:
            raise ValueError("TrainingConfig.game must be set.")
        if self.num_envs < 1:
            raise ValueError("TrainingConfig.num_envs must be at least 1.")
        if self.n_stack < 1:
            raise ValueError("TrainingConfig.n_stack must be at least 1.")
        if self.timestep_batch < 1:
            raise ValueError("TrainingConfig.timestep_batch must be at least 1.")
        if self.population_size < 8:
            raise ValueError("TrainingConfig.population_size must be at least 8.")
