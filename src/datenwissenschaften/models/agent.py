from pathlib import Path
from typing import Any

from loguru import logger
from stable_baselines3 import DQN


def load_agent(environment: Any, path: Path) -> DQN:
    checkpoint = path.with_suffix(".zip")
    if checkpoint.is_file():
        logger.info(f"Loading agent from {checkpoint}")
        return DQN.load(checkpoint, env=environment, device="cpu")
    logger.info("Creating feature-based DQN agent")
    return DQN(
        "MlpPolicy",
        environment,
        device="cpu",
        learning_rate=0.0001,
        buffer_size=100_000,
        learning_starts=5_000,
        batch_size=128,
        train_freq=(1, "step"),
        gradient_steps=1,
        n_steps=3,
        target_update_interval=10_000,
        gamma=0.999,
        exploration_fraction=1.0,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        policy_kwargs={"net_arch": [128, 128]},
    )
