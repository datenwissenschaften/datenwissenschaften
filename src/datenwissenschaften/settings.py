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
    cache_dir: Path


@dataclass(frozen=True)
class TrainingSettings:
    game: str
    game_identity: str
    total_timesteps: int
    savestate: str | None
    savestates: tuple[str, ...]
    num_envs: int

    @property
    def active_savestate(self) -> str | None:
        return self.savestates[0] if self.savestates else self.savestate


@dataclass(frozen=True)
class UploadSettings:
    url: str
    api_key: str | None


@dataclass(frozen=True)
class UISettings:
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 18_080
    max_episodes: int | None = 1_000
    redis_url: str = "redis://127.0.0.1:6379/0"
    history_key_prefix: str = "datenwissenschaften:history"


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

    savestate = _nullable_string(training, "savestate") if "savestate" in training else None
    savestates = _string_tuple(training, "savestates", default=())
    if not savestate and not savestates:
        raise RuntimeError("Missing required configuration value: training.savestate or training.savestates")

    return RetroSpeedlabConfig(
        paths=RetroSpeedlabPaths(
            roms_path=_path(paths, "roms", base_dir),
            models_dir=_path(paths, "models", base_dir),
            record_dir=_path(paths, "recordings", base_dir),
            cache_dir=_path(paths, "cache", base_dir),
        ),
        training=TrainingSettings(
            game=_string(training, "game"),
            game_identity=_optional_string(training, "game_identity") or _string(training, "game"),
            total_timesteps=_positive_int(training, "total_timesteps"),
            savestate=savestate,
            savestates=savestates,
            num_envs=_environment_count(training, "num_envs"),
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
    for path in (paths.models_dir, paths.record_dir, paths.cache_dir):
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


def _string_tuple(values: dict[str, Any], key: str, *, default: tuple[str, ...]) -> tuple[str, ...]:
    value = values.get(key, default)
    if value is None:
        return ()
    if not isinstance(value, list):
        raise RuntimeError(f"Configuration value '{key}' must be a list of non-empty strings.")

    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise RuntimeError(f"Configuration value '{key}' must be a list of non-empty strings.")
        stripped = item.strip()
        if stripped not in result:
            result.append(stripped)
    return tuple(result)


def _positive_int(values: dict[str, Any], key: str, *, default: int | None = None) -> int:
    value = values.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise RuntimeError(f"Configuration value '{key}' must be a positive integer.")
    return value


def _environment_count(values: dict[str, Any], key: str) -> int:
    value = values.get(key)
    if isinstance(value, str) and value.casefold() == "auto":
        return optimal_env_count()
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
    max_episodes = value.get("max_episodes", 1_000)
    redis_url = value.get("redis_url", "redis://127.0.0.1:6379/0")
    history_key_prefix = value.get("history_key_prefix", "datenwissenschaften:history")
    if not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65_535:
        raise RuntimeError("Configuration value 'ui.port' must be between 1 and 65535.")
    if max_episodes is not None and (
        not isinstance(max_episodes, int) or isinstance(max_episodes, bool) or max_episodes < 1
    ):
        raise RuntimeError("Configuration value 'ui.max_episodes' must be null or a positive integer.")
    if not isinstance(redis_url, str) or not redis_url.strip():
        raise RuntimeError("Configuration value 'ui.redis_url' must be a non-empty string.")
    if not isinstance(history_key_prefix, str) or not history_key_prefix.strip():
        raise RuntimeError("Configuration value 'ui.history_key_prefix' must be a non-empty string.")
    return UISettings(
        enabled=enabled,
        host=host.strip(),
        port=port,
        max_episodes=max_episodes,
        redis_url=redis_url.strip(),
        history_key_prefix=history_key_prefix.strip(),
    )
