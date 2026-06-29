from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path("config.yaml")


@dataclass(frozen=True)
class RetroSpeedlabPaths:
    roms_path: Path
    models_dir: Path
    record_dir: Path


@dataclass(frozen=True)
class TrainingSettings:
    game: str
    total_timesteps: int
    savestate: str | None = None
    num_envs: int = 1


@dataclass(frozen=True)
class UploadSettings:
    url: str = "https://speedlab.datenwissenschaften.com/api"
    api_key: str | None = None


@dataclass(frozen=True)
class RetroSpeedlabConfig:
    paths: RetroSpeedlabPaths
    training: TrainingSettings
    log_level: str = "INFO"
    upload: UploadSettings = UploadSettings()


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> RetroSpeedlabConfig:
    config_path = Path(path).expanduser().resolve()
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
    upload = _optional_mapping(document, "upload")
    base_dir = config_path.parent

    return RetroSpeedlabConfig(
        paths=RetroSpeedlabPaths(
            roms_path=_path(paths, "roms", base_dir),
            models_dir=_path(paths, "models", base_dir),
            record_dir=_path(paths, "recordings", base_dir),
        ),
        training=TrainingSettings(
            game=_string(training, "game"),
            total_timesteps=_positive_int(training, "total_timesteps"),
            savestate=_optional_string(training, "savestate"),
            num_envs=_positive_int(training, "num_envs", default=1),
        ),
        log_level=_optional_string(document, "log_level") or "INFO",
        upload=UploadSettings(
            url=_optional_string(upload, "url") or UploadSettings.url,
            api_key=_optional_string(upload, "api_key"),
        ),
    )


def load_paths_from_config(path: str | Path = DEFAULT_CONFIG_PATH) -> RetroSpeedlabPaths:
    return load_config(path).paths


def _mapping(values: dict[str, Any], key: str) -> dict[str, Any]:
    value = values.get(key)
    if not isinstance(value, dict):
        raise RuntimeError(f"Configuration value '{key}' must be a mapping.")
    return value


def _optional_mapping(values: dict[str, Any], key: str) -> dict[str, Any]:
    value = values.get(key, {})
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


def _positive_int(values: dict[str, Any], key: str, *, default: int | None = None) -> int:
    value = values.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise RuntimeError(f"Configuration value '{key}' must be a positive integer.")
    return value


def _path(values: dict[str, Any], key: str, base_dir: Path) -> Path:
    path = Path(_string(values, key)).expanduser()
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()
