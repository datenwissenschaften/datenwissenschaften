from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RNDConfig:
    output_size: int
    learning_rate: float
    update_proportion: float
    initial_coefficient: float
    final_coefficient: float
    anneal_steps: int
    reward_clip: float
    stale_episodes: int
    stale_multiplier: float

    def __post_init__(self) -> None:
        if self.output_size < 1 or self.anneal_steps < 1 or self.stale_episodes < 1:
            raise ValueError("RND sizes, anneal steps, and stale episodes must be positive")
        if self.learning_rate <= 0.0 or not 0.0 < self.update_proportion <= 1.0:
            raise ValueError("RND learning rate and update proportion must be positive")
        if self.initial_coefficient < 0.0 or self.final_coefficient < 0.0:
            raise ValueError("RND coefficients must be non-negative")
        if self.reward_clip <= 0.0 or self.stale_multiplier < 1.0:
            raise ValueError("RND reward clip must be positive and stale multiplier must be at least one")
