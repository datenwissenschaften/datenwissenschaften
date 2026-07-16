from __future__ import annotations

import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from loguru import logger
from stable_baselines3.common.callbacks import BaseCallback

from datenwissenschaften.callbacks import BestEpisodeCallback, EpisodeTelemetryCallback
from datenwissenschaften.callbacks.save_model_callback import atomic_save
from datenwissenschaften.callbacks.upload_episode_callback import UploadEpisodeCallback
from datenwissenschaften.core.protocols import TrainableModel
from datenwissenschaften.model import ModelBuilder, get_model_metadata, get_model_path
from datenwissenschaften.runtime import get_runtime
from datenwissenschaften.segmented_rollout import SegmentedRecurrentRollouts
from datenwissenschaften.settings import DEFAULT_CONFIG_PATH, load_config
from datenwissenschaften.trainer import Trainer
from datenwissenschaften.ui import configure_history, publish_metadata, start_ui
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
        schedule_horizon = 10**18
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
            model._total_timesteps, lifecycle_callbacks[state_name] = model._setup_learn(
                schedule_horizon,
                callback=[],
                reset_num_timesteps=False,
                tb_log_name=f"state_{state_name}",
                progress_bar=False,
            )
            lifecycle_callbacks[state_name].on_training_start(locals(), globals())

        observations = venv.reset()
        savestate_scheduler = SavestateScheduler(
            config.training.savestates,
            interval_seconds=config.training.savestate_rotation_seconds,
        )
        rollouts = SegmentedRecurrentRollouts(models, venv.num_envs)
        updates = 0
        global_steps = 0
        segment_counts = dict.fromkeys(models, 0)
        update_counts = dict.fromkeys(models, 0)
        best_state_fitness: dict[str, float | None] = dict.fromkeys(models)
        self._start_segmented_ui(venv, models, config)
        episode_callbacks = self._start_episode_callbacks(models, config)
        self._publish_state_training(
            models,
            rollouts,
            [],
            segment_counts,
            update_counts,
            best_state_fitness,
            config,
        )

        while True:
            reset_request = consume_model_reset()
            if reset_request is not None:
                for callback in lifecycle_callbacks.values():
                    callback.on_training_end()
                for callback in episode_callbacks:
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
            global_steps += venv.num_envs
            episode_locals = {"rewards": rewards, "dones": dones, "infos": infos}
            for callback in episode_callbacks:
                callback.update_locals(episode_locals)
                callback.on_step()
            enabled_states = set(models)
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

            rotation_reason = savestate_scheduler.rotation_reason(
                won=any(bool(info.get("won")) and bool(info.get("curriculum_complete")) for info in infos),
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
                segment_counts[state_names[env_index]] += 1
                state_fitness = float(info.get("state_return", rewards[env_index]))
                previous_best = best_state_fitness[state_names[env_index]]
                if previous_best is None or state_fitness > previous_best:
                    best_state_fitness[state_names[env_index]] = state_fitness
                record_outcome = getattr(model, "record_episode_outcome", None)
                if callable(record_outcome):
                    record_outcome(
                        fitness=state_fitness,
                        won=None if info.get("won") is None else bool(info["won"]),
                    )

            states_to_update = (
                {name for name, transitions in rollouts.transitions.items() if transitions}
                if rotation_reason is not None
                else full
            )
            for state_name in states_to_update:
                self._update_model(state_name, models[state_name], rollouts)
                updates += 1
                update_counts[state_name] += 1
                if updates % max(1, len(models)) == 0:
                    logger.debug(
                        "State-routed training progress: {}",
                        ", ".join(f"{name}={model.num_timesteps:,}" for name, model in models.items()),
                    )
            if states_to_update:
                for callback in episode_callbacks:
                    callback.on_rollout_end()

            if rotation_reason is not None:
                next_savestate = savestate_scheduler.rotate()
                logger.info(f"Rotating savestate to {next_savestate} after {rotation_reason}.")
                get_runtime().set_savestate(next_savestate)
                venv.env_method("set_initial_savestate", next_savestate)
                observations = venv.reset()
                for env_index in range(venv.num_envs):
                    rollouts.reset_environment(env_index)
                for callback in episode_callbacks:
                    callback.on_training_end()
                episode_callbacks = self._start_episode_callbacks(models, config)
                self._publish_run_savestate(config, models, next_savestate)
                continue
            if config.ui.enabled and (global_steps % 128 < venv.num_envs or full):
                self._publish_state_training(
                    models,
                    rollouts,
                    state_names,
                    segment_counts,
                    update_counts,
                    best_state_fitness,
                    config,
                )
            observations = new_observations

    @staticmethod
    def _publish_run_savestate(config: Any, models: dict[str, TrainableModel], savestate: str) -> None:
        if not config.ui.enabled:
            return
        publish_metadata(
            "run",
            {
                "game": config.training.game,
                "game_identity": config.training.game_identity,
                "training_state": "state-routed",
                "savestate": savestate,
                "savestates": list(config.training.savestates),
                "savestate_rotation_seconds": config.training.savestate_rotation_seconds,
                "configured_envs": config.training.num_envs,
                "state_models": list(models),
            },
            replace=True,
        )

    def _update_model(
        self,
        state_name: str,
        model: TrainableModel,
        rollouts: SegmentedRecurrentRollouts,
    ) -> None:
        model.rollout_buffer = rollouts.build_buffer(state_name)
        model._current_progress_remaining = 1.0
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
    def _publish_state_training(
        models: dict[str, TrainableModel],
        rollouts: SegmentedRecurrentRollouts,
        active_states: list[str],
        segment_counts: dict[str, int],
        update_counts: dict[str, int],
        best_state_fitness: dict[str, float | None],
        config: Any,
    ) -> None:
        if not config.ui.enabled:
            return
        active_counts = Counter(active_states)
        publish_metadata(
            "state_training",
            {
                state_name: {
                    "active_environments": active_counts[state_name],
                    "collected_steps": model.num_timesteps,
                    "rollout_steps": len(rollouts.transitions[state_name]),
                    "rollout_capacity": model.n_steps * model.n_envs,
                    "model_updates": update_counts[state_name],
                    "completed_segments": segment_counts[state_name],
                    "best_fitness": best_state_fitness[state_name],
                }
                for state_name, model in models.items()
            },
            replace=True,
        )

    def _start_episode_callbacks(self, models: dict[str, TrainableModel], config: Any) -> list[BaseCallback]:
        runtime_trainer = Trainer(
            config_path=self.config_path,
            state_name=config.training.active_savestate or "episode",
        )
        runtime_trainer._configure_runtime()
        callbacks: list[BaseCallback] = [
            BestEpisodeCallback(),
            EpisodeTelemetryCallback(),
            UploadEpisodeCallback(config.upload),
        ]
        representative_model = next(iter(models.values()))
        for callback in callbacks:
            callback.init_callback(representative_model)
            callback.on_training_start(locals(), globals())
        return callbacks

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
                "savestate_rotation_seconds": config.training.savestate_rotation_seconds,
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


class SavestateScheduler:
    def __init__(
        self,
        savestates: Sequence[str],
        *,
        interval_seconds: int,
        clock=time.monotonic,
    ) -> None:
        self.savestates = tuple(savestates)
        self.interval_seconds = interval_seconds
        self.clock = clock
        self.index = 0
        self.rotated_at = self.clock()

    def rotation_reason(self, *, won: bool) -> str | None:
        if len(self.savestates) < 2:
            return None
        if won:
            return "successful episode"
        if self.clock() - self.rotated_at >= self.interval_seconds:
            return f"{self.interval_seconds} seconds"
        return None

    def rotate(self) -> str:
        if not self.savestates:
            raise ValueError("Cannot rotate an empty savestate list.")
        self.index = (self.index + 1) % len(self.savestates)
        self.rotated_at = self.clock()
        return self.savestates[self.index]
