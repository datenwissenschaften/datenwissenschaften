from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from datenwissenschaften.parallelism import optimal_env_count

DEFAULT_CONFIG_PATH = Path("config.yaml")


@dataclass(frozen=True)
class RetroSpeedlabPaths:
    roms_path: Path
    models_dir: Path
    record_dir: Path
    savestate_dir: Path


@dataclass(frozen=True)
class TrainingSettings:
    game: str
    total_timesteps: int
    population_size: int
    savestate: str | None
    num_envs: int


@dataclass(frozen=True)
class UploadSettings:
    url: str
    api_key: str | None


@dataclass(frozen=True)
class UISettings:
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 18_080
    max_episodes: int = 5_000


@dataclass(frozen=True)
class RetroSpeedlabConfig:
    paths: RetroSpeedlabPaths
    training: TrainingSettings
    log_level: str
    upload: UploadSettings
    ui: UISettings


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> RetroSpeedlabConfig:
    config_path = Path(config_path).expanduser().resolve()
    try:
        with config_path.open(encoding="utf-8") as config_file:
            document = yaml.safe_load(config_file)
    except FileNotFoundError as error:
        raise RuntimeError(f"Configuration file not found: {config_path}") from error
    except yaml.YAMLError as error:
        raise RuntimeError(f"Invalid YAML in configuration file {config_path}: {error}") from error

    if not isinstance(document, dict):
        raise RuntimeError(f"Configuration file must contain a YAML mapping: {config_path}")

    paths = _mapping(document, "paths")
    training = _mapping(document, "training")
    upload = _mapping(document, "upload")
    base_dir = config_path.parent

    population_size = _positive_int(training, "population_size")

    return RetroSpeedlabConfig(
        paths=RetroSpeedlabPaths(
            roms_path=_path(paths, "roms", base_dir),
            models_dir=_path(paths, "models", base_dir),
            record_dir=_path(paths, "recordings", base_dir),
            savestate_dir=_path(paths, "savestates", base_dir),
        ),
        training=TrainingSettings(
            game=_string(training, "game"),
            total_timesteps=_positive_int(training, "total_timesteps"),
            savestate=_nullable_string(training, "savestate"),
            num_envs=_environment_count(training, "num_envs", population_size),
            population_size=population_size,
        ),
        log_level=_string(document, "log_level"),
        upload=UploadSettings(
            url=_string(upload, "url"),
            api_key=_nullable_string(upload, "api_key"),
        ),
        ui=_ui_settings(document.get("ui")),
    )


def empty_all_paths(config_path: str | Path = DEFAULT_CONFIG_PATH) -> None:
    paths = load_config(config_path).paths
    for path in (paths.models_dir, paths.record_dir, paths.savestate_dir):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)


def load_paths_from_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> RetroSpeedlabPaths:
    return load_config(config_path).paths


def _mapping(values: dict[str, Any], key: str) -> dict[str, Any]:
    value = values.get(key)
    if not isinstance(value, dict):
        raise RuntimeError(f"Configuration value '{key}' must be a mapping.")
    return value


def _string(values: dict[str, Any], key: str) -> str:
    value = _optional_string(values, key)
    if value is None:
        raise RuntimeError(f"Missing required configuration value: {key}")
    return value


def _optional_string(values: dict[str, Any], key: str) -> str | None:
    value = values.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"Configuration value '{key}' must be a non-empty string.")
    return value


def _nullable_string(values: dict[str, Any], key: str) -> str | None:
    if key not in values:
        raise RuntimeError(f"Missing required configuration value: {key}")
    return _optional_string(values, key)


def _positive_int(values: dict[str, Any], key: str) -> int:
    value = values.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise RuntimeError(f"Configuration value '{key}' must be a positive integer.")
    return value


def _environment_count(values: dict[str, Any], key: str, population_size: int) -> int:
    value = values.get(key)
    if isinstance(value, str) and value.casefold() == "auto":
        return optimal_env_count(population_size)
    return _positive_int(values, key)


def _path(values: dict[str, Any], key: str, base_dir: Path) -> Path:
    path = Path(_string(values, key)).expanduser()
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def _ui_settings(value: Any) -> UISettings:
    if value is None:
        return UISettings()
    if isinstance(value, bool):
        return UISettings(enabled=value)
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"enable", "enabled", "on", "true"}:
            return UISettings(enabled=True)
        if normalized in {"disable", "disabled", "off", "false"}:
            return UISettings(enabled=False)
        raise RuntimeError("Configuration value 'ui' must be 'enable' or 'disable'.")
    if not isinstance(value, dict):
        raise RuntimeError("Configuration value 'ui' must be a string, boolean, or mapping.")

    if "enable" in value and "enabled" in value:
        raise RuntimeError("Use only one of 'ui.enable' or the legacy 'ui.enabled' value.")
    enabled = value.get("enable", value.get("enabled", True))
    if not isinstance(enabled, bool):
        raise RuntimeError("Configuration value 'ui.enable' must be a boolean.")
    host = value.get("host", "127.0.0.1")
    if not isinstance(host, str) or not host.strip():
        raise RuntimeError("Configuration value 'ui.host' must be a non-empty string.")
    port = value.get("port", 18_080)
    max_episodes = value.get("max_episodes", 5_000)
    if not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65_535:
        raise RuntimeError("Configuration value 'ui.port' must be between 1 and 65535.")
    if not isinstance(max_episodes, int) or isinstance(max_episodes, bool) or max_episodes < 1:
        raise RuntimeError("Configuration value 'ui.max_episodes' must be a positive integer.")
    return UISettings(enabled=enabled, host=host.strip(), port=port, max_episodes=max_episodes)
