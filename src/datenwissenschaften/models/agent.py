from pathlib import Path
from typing import Any

from loguru import logger
from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure


def load_agent(environment: Any, path: Path) -> DQN:
    checkpoint = path.with_suffix(".zip")
    replay_buffer = path.with_suffix(".replay.pkl")
    if checkpoint.is_file():
        if not replay_buffer.is_file():
            raise RuntimeError(f"Replay buffer not found: {replay_buffer}")
        logger.info(f"Loading agent from {checkpoint}")
        model = DQN.load(checkpoint, env=environment, device="cpu")
        model.load_replay_buffer(replay_buffer, truncate_last_traj=True)
        model.set_logger(configure(folder=None, format_strings=[]))
        return model
    logger.info("Creating feature-based DQN agent")
    model = DQN(
        "MlpPolicy",
        environment,
        device="cpu",
        learning_rate=0.0001,
        buffer_size=100_000,
        learning_starts=5_000,
        batch_size=128,
        train_freq=(1, "step"),
        gradient_steps=1,
        n_steps=1,
        target_update_interval=10_000,
        gamma=0.999,
        exploration_fraction=1.0,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        replay_buffer_kwargs={"handle_timeout_termination": False},
        policy_kwargs={"net_arch": [128, 128]},
    )
    model.set_logger(configure(folder=None, format_strings=[]))
    return model
