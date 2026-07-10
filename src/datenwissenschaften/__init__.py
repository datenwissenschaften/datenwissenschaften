from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "EnvironmentBuilder": ("datenwissenschaften.retro", "EnvironmentBuilder"),
    "GameDefinition": ("datenwissenschaften.retro", "GameDefinition"),
    "GameDefinitionLoader": ("datenwissenschaften.retro", "GameDefinitionLoader"),
    "GameRegistry": ("datenwissenschaften.retro", "GameRegistry"),
    "ModelBuilder": ("datenwissenschaften.model", "ModelBuilder"),
    "PolicyManager": ("datenwissenschaften.policy_manager", "PolicyManager"),
    "RetroEnvironmentFactory": ("datenwissenschaften.retro", "RetroEnvironmentFactory"),
    "RetroSpeedlabConfig": ("datenwissenschaften.settings", "RetroSpeedlabConfig"),
    "RetroSpeedlabPaths": ("datenwissenschaften.retro", "RetroSpeedlabPaths"),
    "RetroSpeedlabRuntime": ("datenwissenschaften.runtime", "RetroSpeedlabRuntime"),
    "RetroVecEnvBuilder": ("datenwissenschaften.retro", "RetroVecEnvBuilder"),
    "AdaptiveRecurrentRNDPPO": ("datenwissenschaften.recurrent_rnd", "AdaptiveRecurrentRNDPPO"),
    "AdaptiveRecurrentRNDModel": ("datenwissenschaften.recurrent_rnd", "AdaptiveRecurrentRNDModel"),
    "SavestateResolver": ("datenwissenschaften.retro", "SavestateResolver"),
    "StateTrainer": ("datenwissenschaften.state_trainer", "StateTrainer"),
    "Trainer": ("datenwissenschaften.trainer", "Trainer"),
    "TrainingConfig": ("datenwissenschaften.core", "TrainingConfig"),
    "TrainingSession": ("datenwissenschaften.core", "TrainingSession"),
    "configure_accelerator": ("datenwissenschaften.accelerator", "configure_accelerator"),
    "configure_runtime": ("datenwissenschaften.runtime", "configure_runtime"),
    "build_adaptive_recurrent_rnd_ppo": (
        "datenwissenschaften.recurrent_rnd",
        "build_adaptive_recurrent_rnd_ppo",
    ),
    "get_model_path": ("datenwissenschaften.model", "get_model_path"),
    "has_boundary_savestate": ("datenwissenschaften.retro.savestates", "has_boundary_savestate"),
    "load_boundary_savestate": ("datenwissenschaften.retro.savestates", "load_boundary_savestate"),
    "save_boundary_savestate": ("datenwissenschaften.retro.savestates", "save_boundary_savestate"),
    "import_roms": ("datenwissenschaften.roms", "import_roms"),
    "load_config": ("datenwissenschaften.settings", "load_config"),
    "load_or_create_model": ("datenwissenschaften.model", "load_or_create_model"),
    "load_paths_from_config": ("datenwissenschaften.settings", "load_paths_from_config"),
    "optimal_env_count": ("datenwissenschaften.parallelism", "optimal_env_count"),
    "setup_logging": ("datenwissenschaften.logger", "setup_logging"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error

    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
