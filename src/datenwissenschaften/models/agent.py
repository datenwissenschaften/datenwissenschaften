from pathlib import Path
from typing import Any

from loguru import logger

from datenwissenschaften.models.adaptive_recurrent import AdaptiveRecurrentPPO
from datenwissenschaften.models.rnd_config import RNDConfig

RND_CONFIG = RNDConfig(
    output_size=128,
    learning_rate=0.0001,
    update_proportion=0.25,
    initial_coefficient=0.25,
    final_coefficient=0.01,
    anneal_steps=5_000_000,
    reward_clip=1.0,
    stale_episodes=32,
    stale_multiplier=2.0,
)


def load_agent(environment: Any, path: Path) -> AdaptiveRecurrentPPO:
    checkpoint = path.with_suffix(".zip")
    if checkpoint.is_file():
        logger.info(f"Loading agent from {checkpoint}")
        return AdaptiveRecurrentPPO.load(checkpoint, env=environment, device="cpu")
    logger.info("Creating adaptive recurrent PPO agent with RND")
    model = AdaptiveRecurrentPPO(
        "MultiInputLstmPolicy",
        environment,
        device="cpu",
        learning_rate=0.0002,
        n_steps=512,
        batch_size=256,
        n_epochs=4,
        gamma=0.999,
        gae_lambda=0.98,
        ent_coef=0.01,
        policy_kwargs={
            "lstm_hidden_size": 256,
            "n_lstm_layers": 1,
            "shared_lstm": False,
            "enable_critic_lstm": True,
        },
    )
    model.configure_rnd(RND_CONFIG)
    return model
