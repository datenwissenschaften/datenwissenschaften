import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

from loguru import logger
from stable_baselines3.common.callbacks import BaseCallback, ConvertCallback

from datenwissenschaften.checkpoints.model import atomic_save
from datenwissenschaften.configuration.loader import load_config
from datenwissenschaften.models.agent import load_agent
from datenwissenschaften.models.path import curriculum_model_directory, model_directory
from datenwissenschaften.rewards.normalizer import save_reward_normalizer
from datenwissenschaften.training.runner_stats import RunnerStatsPublisher
from datenwissenschaften.training.winning_episode_uploader import WinningEpisodeUploader

CHECKPOINT_INTERVAL = 10_000


def train(environment: Any, config_path: str | Path) -> None:
    config = load_config(config_path)
    logger.remove()
    logger.add(sys.stderr, level=config.log_level)
    state_names = _state_names(environment)
    models = {
        state_name: load_agent(
            environment,
            curriculum_model_directory(config, state_name) / "model",
        )
        for state_name in state_names
    }
    uploader = WinningEpisodeUploader(config)
    stats_publisher = RunnerStatsPublisher(config)
    logger.info(
        "Training {} curriculum model(s) for {} / {} with {} environment(s)",
        len(models),
        config.training.game,
        config.training.savestate,
        environment.num_envs,
    )

    while True:
        state_name = _active_state(environment, state_names)
        model = models[state_name]
        checkpoint = curriculum_model_directory(config, state_name) / "model"
        model.set_env(environment, force_reset=True)
        model.learn(
            total_timesteps=CHECKPOINT_INTERVAL,
            reset_num_timesteps=False,
            callback=[
                uploader,
                stats_publisher,
                ConvertCallback(_checkpoint_callback(model, checkpoint, model_directory(config))),
                CurriculumSwitchCallback(state_name),
            ],
        )
        if uploader.completed:
            uploader.remove_model()
            logger.success("Removed completed agent")
            return
        if uploader.upload_failed:
            raise RuntimeError("Training stopped because an episode upload failed")
        _save_model(model, checkpoint, model_directory(config))
        environment.reset()
        if _curriculum_complete(environment):
            _run_full_runs(environment, models, uploader, stats_publisher)
            if uploader.completed:
                uploader.remove_model()
                logger.success("Removed completed agent")
                return


def _state_names(environment: Any) -> tuple[str, ...]:
    state_types = tuple(environment.get_attr("state_types")[0])
    names = tuple(state_type.__name__ for state_type in state_types)
    if not names:
        raise RuntimeError("Training requires at least one curriculum state")
    if len(names) != len(set(names)):
        raise RuntimeError("Curriculum state names must be unique")
    return names


def _active_state(environment: Any, state_names: tuple[str, ...]) -> str:
    states = tuple(environment.get_attr("curriculum_state"))
    if not states or any(state != states[0] for state in states):
        raise RuntimeError("All environments must share one active curriculum state")
    state = states[0]
    if state not in state_names:
        raise RuntimeError(f"Environment returned unknown curriculum state: {state}")
    return state


def _checkpoint_callback(
    model: Any,
    checkpoint: Path,
    normalizer_directory: Path,
) -> Callable[[dict[str, Any], dict[str, Any]], bool]:
    next_checkpoint = [
        (model.num_timesteps // CHECKPOINT_INTERVAL + 1) * CHECKPOINT_INTERVAL,
    ]
    return partial(
        _save_checkpoint,
        checkpoint=checkpoint,
        normalizer_directory=normalizer_directory,
        next_checkpoint=next_checkpoint,
    )


def _save_checkpoint(
    locals_: dict[str, Any],
    globals_: dict[str, Any],
    *,
    checkpoint: Path,
    normalizer_directory: Path,
    next_checkpoint: list[int],
) -> bool:
    del globals_
    model = locals_["self"]
    if model.num_timesteps < next_checkpoint[0]:
        return True

    _save_model(model, checkpoint, normalizer_directory)
    logger.debug("Saved curriculum agent after {:,} environment steps", model.num_timesteps)
    next_checkpoint[0] = (model.num_timesteps // CHECKPOINT_INTERVAL + 1) * CHECKPOINT_INTERVAL
    return True


def _save_model(model: Any, checkpoint: Path, normalizer_directory: Path) -> None:
    atomic_save(model, checkpoint)
    save_reward_normalizer(model.get_env(), normalizer_directory)


def _curriculum_complete(environment: Any) -> bool:
    curriculum = environment.get_attr("curriculum")[0]
    return curriculum.is_complete()


def _run_full_runs(
    environment: Any,
    models: dict[str, Any],
    uploader: WinningEpisodeUploader,
    stats_publisher: RunnerStatsPublisher,
) -> None:
    observations = environment.reset()
    while not uploader.completed and not uploader.upload_failed:
        state_names = tuple(environment.get_attr("curriculum_state"))
        if not state_names or any(state != state_names[0] for state in state_names):
            raise RuntimeError("All environments must share one full-run curriculum state")
        state_name = state_names[0]
        model = models[state_name]
        actions, _ = model.predict(observations, deterministic=True)
        observations, _, dones, infos = environment.step(actions)
        stats_publisher.process(dones, infos)
        uploader.process(dones, infos)
    if uploader.upload_failed:
        raise RuntimeError("Full run stopped because an episode upload failed")


class CurriculumSwitchCallback(BaseCallback):
    def __init__(self, state_name: str) -> None:
        super().__init__()
        self.state_name = state_name

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        states = {info["curriculum_state"] for info in infos}
        if not states:
            raise RuntimeError("Environment returned no curriculum state")
        return states == {self.state_name}
