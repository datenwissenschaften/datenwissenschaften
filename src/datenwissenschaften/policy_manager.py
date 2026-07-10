from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from stable_baselines3 import PPO

from datenwissenschaften.model import ModelLoader, get_model_path
from datenwissenschaften.settings import DEFAULT_CONFIG_PATH, load_config


class PolicyManager:
    """Routes predictions to the sub-model responsible for the current state.

    The state machine keeps deciding the transitions; the manager mirrors them on
    the policy side by selecting the model registered for the active state name
    (as reported by ``info["state"]`` or the wrapper's ``state_name()`` method).
    """

    def __init__(self, models: Mapping[str, Any]) -> None:
        if not models:
            raise ValueError("PolicyManager requires at least one model.")
        self.models = dict(models)

    def model_for(self, state_name: str) -> Any:
        model = self.models.get(state_name)
        if model is None:
            raise KeyError(f"No model registered for state: {state_name}")
        return model

    def predict(self, observation: Any, *, state_name: str, **kwargs) -> Any:
        return self.model_for(state_name).predict(observation, **kwargs)

    def state_names(self) -> list[str]:
        return list(self.models)

    @classmethod
    def load(
        cls,
        state_names: Sequence[str],
        *,
        load_model: ModelLoader = PPO.load,
        env: Any = None,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
    ) -> "PolicyManager":
        config = load_config(config_path)
        models: dict[str, Any] = {}
        for state_name in state_names:
            model_path = get_model_path(
                str(config.paths.models_dir),
                config.training.game_identity,
                state_name,
            )
            models[state_name] = load_model(model_path, env=env)
        return cls(models)
