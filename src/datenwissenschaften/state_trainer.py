from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from loguru import logger
from stable_baselines3.common.callbacks import BaseCallback

from datenwissenschaften.callbacks.save_model_callback import atomic_save
from datenwissenschaften.core.protocols import TrainableModel
from datenwissenschaften.model import ModelBuilder, get_model_metadata, get_model_path
from datenwissenschaften.segmented_rollout import SegmentedRecurrentRollouts
from datenwissenschaften.settings import DEFAULT_CONFIG_PATH, load_config
from datenwissenschaften.trainer import Trainer
from datenwissenschaften.ui import configure_history, publish_episode, publish_metadata, start_ui
from datenwissenschaften.ui.control import (
    configure_training_control,
    consume_model_reset,
    perform_model_reset,
)


class StateTrainer:
    """Trains dedicated state models from the configured level start.

    Every vector environment is routed independently to the model for its active
    state. State transitions cut that model's reward/GAE/LSTM sequence, but never
    reset or terminate the underlying level. Each model is updated only from its
    own compacted transitions.
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

        venv.env_method("set_terminate_on_transition", False)
        venv.env_method("set_transition_bonus", self.transition_bonus)

        models: dict[str, TrainableModel] = {}
        for state_name in state_names:
            logger.info(f"Loading model for state-routed training: {state_name}")
            models[state_name] = self.model_builder.build(venv, state_name=state_name)

        return self._train_segmented(venv, models)

    def _train_segmented(self, venv: Any, models: dict[str, TrainableModel]) -> dict[str, TrainableModel]:
        config = load_config(self.config_path)
        total_timesteps = config.training.total_timesteps
        if self._additional_callbacks:
            logger.warning("Additional callbacks are not yet supported by state-segmented training.")
        lifecycle_callbacks: dict[str, Any] = {}
        for state_name, model in models.items():
            required = ("policy", "rollout_buffer", "n_steps", "n_envs", "train", "_last_lstm_states")
            missing = [name for name in required if not hasattr(model, name)]
            if missing:
                raise TypeError(
                    f"State-segmented training requires a recurrent on-policy model; "
                    f"{state_name} is missing: {', '.join(missing)}"
                )
            remaining = max(0, total_timesteps - model.num_timesteps)
            model._total_timesteps, lifecycle_callbacks[state_name] = model._setup_learn(
                remaining,
                callback=[],
                reset_num_timesteps=False,
                tb_log_name=f"state_{state_name}",
                progress_bar=False,
            )
            lifecycle_callbacks[state_name].on_training_start(locals(), globals())

        observations = venv.reset()
        rollouts = SegmentedRecurrentRollouts(models, venv.num_envs)
        updates = 0
        segment_started_at = [time.monotonic()] * venv.num_envs
        self._start_segmented_ui(venv, models, config)

        while any(
            model.num_timesteps < total_timesteps or bool(rollouts.transitions[state_name])
            for state_name, model in models.items()
        ):
            reset_request = consume_model_reset()
            if reset_request is not None:
                for callback in lifecycle_callbacks.values():
                    callback.on_training_end()
                perform_model_reset(reset_request)
                fresh_models: dict[str, TrainableModel] = {}
                for state_name in models:
                    logger.info(f"Creating fresh model after UI reset: {state_name}")
                    fresh_models[state_name] = self.model_builder.build(venv, state_name=state_name)
                return self._train_segmented(venv, fresh_models)

            state_names = [str(name) for name in venv.env_method("state_name")]
            unknown = sorted(set(state_names) - set(models))
            if unknown:
                raise KeyError(f"No model registered for active state(s): {', '.join(unknown)}")

            actions, decisions = rollouts.actions(observations, state_names)
            new_observations, rewards, dones, infos = venv.step(actions)
            enabled_states = {
                state_name
                for state_name, model in models.items()
                if model.num_timesteps < total_timesteps or bool(rollouts.transitions[state_name])
            }
            full = rollouts.append(
                observations,
                actions,
                rewards,
                new_observations,
                dones,
                infos,
                state_names,
                decisions,
                enabled_states,
            )

            for state_name in state_names:
                if state_name in enabled_states:
                    models[state_name].num_timesteps += 1
            for env_index, info in enumerate(infos):
                if not info.get("state_segment_end"):
                    continue
                if state_names[env_index] not in enabled_states:
                    continue
                model = models[state_names[env_index]]
                if config.ui.enabled:
                    state_steps = int(info.get("state_steps", 0))
                    publish_episode(
                        env=env_index,
                        training_state=state_names[env_index],
                        fitness=float(info.get("state_return", rewards[env_index])),
                        training_steps=state_steps,
                        total_steps=state_steps,
                        duration_seconds=time.monotonic() - segment_started_at[env_index],
                        won=None if info.get("won") is None else bool(info["won"]),
                        final_state=info.get("state"),
                    )
                segment_started_at[env_index] = time.monotonic()
                record_outcome = getattr(model, "record_episode_outcome", None)
                if callable(record_outcome):
                    record_outcome(
                        fitness=float(info.get("state_return", rewards[env_index])),
                        won=None if info.get("won") is None else bool(info["won"]),
                    )

            for state_name in full:
                self._update_model(state_name, models[state_name], rollouts, total_timesteps)
                updates += 1
                if updates % max(1, len(models)) == 0:
                    logger.debug(
                        "State-routed training progress: {}",
                        ", ".join(f"{name}={model.num_timesteps:,}" for name, model in models.items()),
                    )
            observations = new_observations

        for callback in lifecycle_callbacks.values():
            callback.on_training_end()
        return models

    def _update_model(
        self,
        state_name: str,
        model: TrainableModel,
        rollouts: SegmentedRecurrentRollouts,
        total_timesteps: int,
    ) -> None:
        model.rollout_buffer = rollouts.build_buffer(state_name)
        model._update_current_progress_remaining(model.num_timesteps, total_timesteps)
        configured_batch_size = model.batch_size
        model.batch_size = model.rollout_buffer.buffer_size
        try:
            model.train()
        finally:
            model.batch_size = configured_batch_size
        model_path = get_model_path(
            str((config := load_config(self.config_path)).paths.models_dir),
            config.training.game_identity,
            state_name,
        )
        atomic_save(model, model_path)

    @staticmethod
    def _start_segmented_ui(venv: Any, models: dict[str, TrainableModel], config: Any) -> None:
        if not config.ui.enabled:
            return
        configure_history(
            config.training.game_identity,
            redis_url=config.ui.redis_url,
            key_prefix=config.ui.history_key_prefix,
        )
        configure_training_control(
            game=config.training.game,
            model_dir=config.paths.models_dir,
            restart_supported=True,
            artifact_dirs=(
                config.paths.models_dir,
                config.paths.record_dir,
                config.paths.cache_dir,
            ),
        )
        if start_ui(config.ui) is None:
            return
        publish_metadata(
            "run",
            {
                "game": config.training.game,
                "game_identity": config.training.game_identity,
                "training_state": "state-routed",
                "savestate": config.training.active_savestate,
                "savestates": list(config.training.savestates),
                "total_timesteps": config.training.total_timesteps,
                "configured_envs": config.training.num_envs,
                "state_models": list(models),
            },
            replace=True,
        )
        publish_metadata(
            "models",
            {state_name: get_model_metadata(model) for state_name, model in models.items()},
            replace=True,
        )
        publish_metadata("model", get_model_metadata(next(iter(models.values()))), replace=True)
        publish_metadata("environment", Trainer._environment_metadata(venv), replace=True)
