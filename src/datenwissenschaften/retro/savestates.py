from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger

from datenwissenschaften.persistence import RedisStore
from datenwissenschaften.settings import load_config


def _store_and_key(game: str, state_name: str) -> tuple[RedisStore, tuple[str, ...]]:
    config = load_config()
    return RedisStore(config.ui.redis_url), ("boundary-savestate", game, state_name)


def has_boundary_savestate(game: str, state_name: str) -> bool:
    store, key = _store_and_key(game, state_name)
    return store.get_bytes(*key) is not None


def save_boundary_savestate(env: Any, state_name: str) -> str:
    emulation = env.unwrapped
    store, key = _store_and_key(emulation.gamename, state_name)
    redis_key = store.key(*key)
    if store.get_bytes(*key) is not None:
        return redis_key

    store.set_bytes(*key, value=emulation.em.get_state())
    logger.info(f"Saved boundary savestate for state {state_name} in Redis: {redis_key}")
    return redis_key


def load_boundary_savestate(env: Any, state_name: str) -> np.ndarray:
    emulation = env.unwrapped
    store, key = _store_and_key(emulation.gamename, state_name)
    data = store.get_bytes(*key)
    if data is None:
        raise FileNotFoundError(f"No boundary savestate in Redis for state {state_name}")

    emulation.em.set_state(data)
    emulation.data.reset()
    emulation.data.update_ram()
    logger.debug(f"Loaded boundary savestate for state {state_name} from Redis")
    return emulation.em.get_screen()
