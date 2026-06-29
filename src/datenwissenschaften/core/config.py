from __future__ import annotations

from dataclasses import dataclass

from datenwissenschaften.parallelism import optimal_env_count


@dataclass(frozen=True)
class TrainingConfig:
    game: str
    num_envs: int | str
    n_stack: int
    population_size: int
    timestep_batch: int = int(1e7)

    def __post_init__(self) -> None:
        if not self.game:
            raise ValueError("TrainingConfig.game must be set.")
        if isinstance(self.num_envs, str) and self.num_envs.casefold() == "auto":
            object.__setattr__(self, "num_envs", optimal_env_count(self.population_size))
        if not isinstance(self.num_envs, int) or isinstance(self.num_envs, bool) or self.num_envs < 1:
            raise ValueError("TrainingConfig.num_envs must be at least 1.")
        if self.n_stack < 1:
            raise ValueError("TrainingConfig.n_stack must be at least 1.")
        if self.timestep_batch < 1:
            raise ValueError("TrainingConfig.timestep_batch must be at least 1.")
        if self.population_size < 8:
            raise ValueError("TrainingConfig.population_size must be at least 8.")
