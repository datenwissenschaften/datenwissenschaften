from pathlib import Path
from typing import Any

from loguru import logger
from sb3_contrib import RecurrentPPO


def load_agent(environment: Any, path: Path) -> RecurrentPPO:
    checkpoint = path.with_suffix(".zip")
    if checkpoint.is_file():
        logger.info(f"Loading agent from {checkpoint}")
        return RecurrentPPO.load(checkpoint, env=environment)
    logger.info("Creating recurrent multi-input agent")
    return RecurrentPPO("MultiInputLstmPolicy", environment)
