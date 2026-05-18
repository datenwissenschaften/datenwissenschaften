from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import find_dotenv, load_dotenv
except ModuleNotFoundError:
    find_dotenv = None
    load_dotenv = None

from datenwissenschaften.retro.paths import RetroArenaPaths


def _required_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment value: {name}")
    return Path(value)


def load_paths_from_env() -> RetroArenaPaths:
    if load_dotenv and find_dotenv:
        load_dotenv(find_dotenv(), override=True)

    roms_path = _required_path("RETRO_ARENA_ROM_PATH")
    models_dir = _required_path("RETRO_ARENA_MODELS_DIR")
    working_dir = _required_path("RETRO_ARENA_WORKING_DIR")
    record_dir = _required_path("RETRO_ARENA_RECORDING_DIR")

    return RetroArenaPaths(
        roms_path=roms_path,
        models_dir=models_dir,
        working_dir=working_dir,
        record_dir=record_dir
    )
