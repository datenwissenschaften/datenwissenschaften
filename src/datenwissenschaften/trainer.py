from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback

from datenwissenschaften.callbacks import (
    BestEpisodeCallback,
    SaveModelCallback,
    StopTrainingAtTimestepsCallback,
)
from datenwissenschaften.callbacks.upload_episode_callback import UploadEpisodeCallback
from datenwissenschaften.model import get_model_metadata, get_model_path
from datenwissenschaften.retro.environment import get_last_environment_wrapper
from datenwissenschaften.retro.paths import RetroSpeedlabPaths
from datenwissenschaften.runtime import RetroSpeedlabRuntime, configure_runtime


class Trainer:
    # noinspection PyTypeChecker
    def __init__(
        self,
        *,
        additional_callbacks: Sequence[BaseCallback] | None = None,
    ) -> None:
        self.total_timesteps = int(os.environ.get("RETRO_SPEEDLAB_TIMESTEPS"))
        self.callbacks = self._default_callbacks() + (additional_callbacks or [])
        self._state: dict[str, Any] = {}
        self._savestate = os.environ.get("RETRO_SPEEDLAB_SAVESTATE")

    def train(self, model) -> None:
        self._configure_runtime()
        if model.num_timesteps >= self.total_timesteps:
            return

        model.learn(
            total_timesteps=self.total_timesteps - model.num_timesteps,
            callback=self.callbacks,
            reset_num_timesteps=False,
        )

    def _default_callbacks(self) -> list[BaseCallback]:
        return [
            SaveModelCallback(),
            BestEpisodeCallback(self.total_timesteps),
            UploadEpisodeCallback(),
            StopTrainingAtTimestepsCallback(self.total_timesteps),
        ]

    def _configure_runtime(self) -> None:
        roms_dir = self._required_env("RETRO_SPEEDLAB_ROM_PATH")
        models_dir = self._required_env("RETRO_SPEEDLAB_MODEL_DIR")
        record_dir = self._required_env("RETRO_SPEEDLAB_RECORDING_DIR")
        game = self._required_env("RETRO_SPEEDLAB_GAME_ID")
        wrapper = get_last_environment_wrapper()

        # noinspection PyTypeChecker
        configure_runtime(
            RetroSpeedlabRuntime(
                paths=RetroSpeedlabPaths(
                    roms_path=Path(roms_dir),
                    models_dir=Path(models_dir),
                    working_dir=Path(record_dir),
                    record_dir=Path(record_dir),
                ),
                wrappers={game: wrapper} if wrapper else {},
                ignored_states={game: set()},
                default_states={game: self._savestate or ""},
                obs_size=(96, 96),
                action_repeat=1,
                get_game=lambda: game,
                get_savestate=lambda: self._savestate or "",
                set_savestate=self._set_savestate,
                get_state_value=self._get_state_value,
                set_state_value=self._set_state_value,
                get_model_path=lambda selected_game: get_model_path(models_dir, selected_game),
                get_model_metadata=get_model_metadata,
            )
        )

    def _get_state_value(self, name: str) -> str:
        state_path = self._state_path(name)
        if os.path.exists(state_path):
            with open(state_path, encoding="utf-8") as file:
                return file.read()
        return str(self._state.get(name, "False"))

    def _set_state_value(self, name: str, value: Any) -> None:
        self._state[name] = value
        state_path = self._state_path(name)
        os.makedirs(os.path.dirname(state_path), exist_ok=True)
        with open(state_path, "w", encoding="utf-8") as file:
            file.write(str(value))

    def _set_savestate(self, savestate: str) -> None:
        self._savestate = savestate

    def _state_path(self, name: str) -> str:
        return os.path.join(
            self._required_env("RETRO_SPEEDLAB_MODEL_DIR"),
            self._required_env("RETRO_SPEEDLAB_GAME_ID"),
            self._savestate or "",
            f"{name}.txt",
        )

    @staticmethod
    def _required_env(name: str) -> str:
        value = os.environ.get(name)
        if value is None:
            raise RuntimeError(f"{name} must be set.")
        return value
