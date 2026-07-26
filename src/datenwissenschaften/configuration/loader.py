import os
from pathlib import Path
from typing import Any

import yaml
from box import Box

DEFAULT_CONFIG_PATH = Path("config.yaml")


def load_config(config_path: str | Path) -> Box:
    path = Path(config_path).expanduser().resolve()
    try:
        with path.open(encoding="utf-8") as stream:
            document = yaml.safe_load(stream)
    except FileNotFoundError as error:
        raise RuntimeError(f"Configuration file not found: {path}") from error
    except yaml.YAMLError as error:
        raise RuntimeError(f"Invalid configuration file {path}: {error}") from error
    if not isinstance(document, dict):
        raise RuntimeError(f"Configuration file must contain a YAML mapping: {path}")
    config = Box(document)
    _require(
        config,
        "paths.roms",
        "paths.models",
        "training.game",
        "training.savestate",
        "training.num_envs",
        "upload.url",
        "upload.api_key",
        "log_level",
    )
    config.paths.roms = _path(config.paths.roms, path.parent)
    config.paths.models = _path(config.paths.models, path.parent)
    config.training.game = _string(config.training.game, "training.game")
    config.training.savestate = _string(config.training.savestate, "training.savestate")
    config.training.num_envs = _environment_count(config.training.num_envs)
    config.upload.url = _string(config.upload.url, "upload.url").rstrip("/")
    config.upload.api_key = _string(config.upload.api_key, "upload.api_key")
    config.log_level = _string(config.log_level, "log_level")
    return config


def _require(config: Box, *names: str) -> None:
    for name in names:
        value: Any = config
        for part in name.split("."):
            if not isinstance(value, Box) or part not in value:
                raise RuntimeError(f"Missing required configuration value: {name}")
            value = value[part]
        if value is None or isinstance(value, str) and not value.strip():
            raise RuntimeError(f"Configuration value '{name}' must not be empty")


def _path(value: Any, base_dir: Path) -> Path:
    if not isinstance(value, str):
        raise RuntimeError("Configured paths must be strings")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"Configuration value '{name}' must be a non-empty string")
    return value.strip()


def _environment_count(value: Any) -> int:
    if value == "auto":
        return len(os.sched_getaffinity(0))
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise RuntimeError("training.num_envs must be a positive integer or 'auto'")
    return value
