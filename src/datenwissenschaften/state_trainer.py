from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from loguru import logger
from stable_baselines3.common.callbacks import BaseCallback

from datenwissenschaften.core.protocols import TrainableModel
from datenwissenschaften.model import ModelBuilder
from datenwissenschaften.settings import DEFAULT_CONFIG_PATH
from datenwissenschaften.trainer import Trainer


class StateTrainer:
    """Trains one dedicated model per state, sequentially (options-style hierarchical RL).

    The vectorized environment is reconfigured for each state: episodes start at the
    state's boundary savestate (captured automatically on state transitions), and a
    transition out of the state terminates the episode with a completion bonus so
    every sub-model learns from clean, local episodes.
    """

    def __init__(
        self,
        model_builder: ModelBuilder,
        *,
        transition_bonus: float,
        additional_callbacks: Sequence[BaseCallback] | None = None,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
    ) -> None:
        self.model_builder = model_builder
        self.transition_bonus = transition_bonus
        self._additional_callbacks = list(additional_callbacks or [])
        self.config_path = config_path

    def train(self, venv: Any) -> dict[str, TrainableModel]:
        state_names = list(venv.env_method("training_state_names")[0])
        if not state_names:
            raise ValueError("No training states configured.")

        venv.env_method("set_capture_boundary_savestates", True)
        venv.env_method("set_terminate_on_transition", True)
        venv.env_method("set_transition_bonus", self.transition_bonus)

        models: dict[str, TrainableModel] = {}
        for state_name in state_names:
            logger.info(f"Training model for state: {state_name}")
            venv.env_method("set_active_training_state", state_name)
            venv.reset()

            model = self.model_builder.build(venv, state_name=state_name)
            trainer = Trainer(
                additional_callbacks=self._additional_callbacks,
                config_path=self.config_path,
                state_name=state_name,
            )
            trainer.train(model)

            models[state_name] = model

        return models
