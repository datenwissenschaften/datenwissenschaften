from __future__ import annotations

import gzip
import os
from pathlib import Path
from typing import Any

import numpy as np
import stable_retro as retro
from loguru import logger


def boundary_savestate_path(game: str, state_name: str) -> Path:
    rom_path = retro.data.get_romfile_path(game)
    return Path(rom_path).parent / f"{state_name}.state"


def has_boundary_savestate(game: str, state_name: str) -> bool:
    return boundary_savestate_path(game, state_name).exists()


def save_boundary_savestate(env: Any, state_name: str) -> Path:
    emulation = env.unwrapped
    path = boundary_savestate_path(emulation.gamename, state_name)
    if path.exists():
        return path

    data = emulation.em.get_state()
    temporary_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with gzip.open(temporary_path, "wb") as file:
        file.write(data)
    temporary_path.replace(path)
    logger.info(f"Saved boundary savestate for state {state_name}: {path}")
    return path


def load_boundary_savestate(env: Any, state_name: str) -> np.ndarray:
    emulation = env.unwrapped
    path = boundary_savestate_path(emulation.gamename, state_name)
    if not path.exists():
        raise FileNotFoundError(f"No boundary savestate for state {state_name}: {path}")

    with gzip.open(path, "rb") as file:
        data = file.read()

    emulation.em.set_state(data)
    emulation.data.reset()
    emulation.data.update_ram()
    logger.debug(f"Loaded boundary savestate for state {state_name}: {path}")
    return emulation.em.get_screen()
