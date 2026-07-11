from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
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
        self._recurrent_states: dict[str, Any] = {}
        self._model_started: set[str] = set()

    def model_for(self, state_name: str) -> Any:
        model = self.models.get(state_name)
        if model is None:
            raise KeyError(f"No model registered for state: {state_name}")
        return model

    def predict(self, observation: Any, *, state_name: str, **kwargs) -> Any:
        """Predict with the active state's policy without resetting the episode.

        Recurrent state is stored independently for every policy. The first call
        to a policy in an episode is marked as an episode start; switching back to
        a previously used policy restores its own LSTM state.
        """
        model = self.model_for(state_name)
        kwargs.setdefault("state", self._recurrent_states.get(state_name))
        kwargs.setdefault(
            "episode_start",
            np.asarray([state_name not in self._model_started], dtype=bool),
        )
        prediction = model.predict(observation, **kwargs)
        self._model_started.add(state_name)
        if isinstance(prediction, tuple) and len(prediction) == 2:
            self._recurrent_states[state_name] = prediction[1]
        return prediction

    def predict_for_env(self, observation: Any, env: Any, **kwargs) -> Any:
        """Route a prediction using the environment's current state."""
        return self.predict(observation, state_name=self._state_name(env), **kwargs)

    def reset_episode(self) -> None:
        """Discard all policy memory after the environment episode resets."""
        self._recurrent_states.clear()
        self._model_started.clear()

    def state_names(self) -> list[str]:
        return list(self.models)

    @staticmethod
    def _state_name(env: Any) -> str:
        state_name = getattr(env, "state_name", None)
        if callable(state_name):
            return str(state_name())

        env_method = getattr(env, "env_method", None)
        if callable(env_method):
            values = env_method("state_name")
            if len(values) != 1:
                raise ValueError("PolicyManager.predict_for_env requires a single environment.")
            return str(values[0])

        raise TypeError("Environment does not expose state_name() or env_method('state_name').")

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
