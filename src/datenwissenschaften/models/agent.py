from pathlib import Path
from typing import Any

from loguru import logger
from stable_baselines3 import PPO


def load_agent(environment: Any, path: Path) -> PPO:
    checkpoint = path.with_suffix(".zip")
    if checkpoint.is_file():
        logger.info(f"Loading agent from {checkpoint}")
        return PPO.load(checkpoint, env=environment, device="cpu")
    logger.info("Creating visual and RAM PPO agent")
    return PPO("MultiInputPolicy", environment, device="cpu")
