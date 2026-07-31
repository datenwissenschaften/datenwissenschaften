from pathlib import Path
from typing import Any

from loguru import logger
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.policies import RecurrentMultiInputActorCriticPolicy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.save_util import load_from_zip_file

from datenwissenschaften.rewards.normalizer import REWARD_DISCOUNT_FACTOR


def load_agent(environment: Any, path: Path) -> RecurrentPPO:
    checkpoint = path.with_suffix(".zip")
    if checkpoint.is_file():
        _validate_checkpoint(checkpoint)
        logger.info(f"Loading agent from {checkpoint}")
        model = RecurrentPPO.load(checkpoint, env=environment, device="auto")
        model.set_logger(configure(folder=None, format_strings=[]))
        return model
    logger.info("Creating recurrent visual-state PPO agent")
    model = RecurrentPPO(
        "MultiInputLstmPolicy",
        environment,
        device="auto",
        learning_rate=0.00025,
        n_steps=256,
        batch_size=256,
        n_epochs=4,
        gamma=REWARD_DISCOUNT_FACTOR,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.03,
        policy_kwargs={
            "lstm_hidden_size": 128,
            "n_lstm_layers": 1,
            "shared_lstm": True,
            "enable_critic_lstm": False,
            "net_arch": {"pi": [64], "vf": [64]},
            "features_extractor_kwargs": {"cnn_output_dim": 128},
        },
    )
    model.set_logger(configure(folder=None, format_strings=[]))
    return model


def _validate_checkpoint(checkpoint: Path) -> None:
    data, _, _ = load_from_zip_file(checkpoint, device="cpu")
    if "policy_class" not in data:
        raise RuntimeError(f"Invalid checkpoint: {checkpoint}")
    policy_class = data["policy_class"]
    if not isinstance(policy_class, type) or not issubclass(policy_class, RecurrentMultiInputActorCriticPolicy):
        raise RuntimeError(f"Unsupported checkpoint algorithm: {checkpoint}")
