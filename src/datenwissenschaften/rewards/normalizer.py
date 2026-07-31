import tempfile
from pathlib import Path

from loguru import logger
from stable_baselines3.common.vec_env import VecEnv, VecNormalize

REWARD_DISCOUNT_FACTOR = 0.995
REWARD_CLIP = 10.0
REWARD_NORMALIZER_FILE = "reward_normalizer.pkl"


def normalize_rewards(environment: VecEnv, model_dir: Path) -> VecNormalize:
    path = model_dir / REWARD_NORMALIZER_FILE
    if path.is_file():
        logger.info("Loading reward normalization statistics from {}", path)
        return VecNormalize.load(str(path), environment)
    return VecNormalize(
        environment,
        norm_obs=False,
        norm_reward=True,
        clip_reward=REWARD_CLIP,
        gamma=REWARD_DISCOUNT_FACTOR,
    )


def save_reward_normalizer(environment: VecNormalize, model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    target = model_dir / REWARD_NORMALIZER_FILE
    with tempfile.TemporaryDirectory(dir=model_dir) as directory:
        temporary = Path(directory) / REWARD_NORMALIZER_FILE
        environment.save(str(temporary))
        temporary.replace(target)


def remove_reward_normalizer(model_dir: Path) -> None:
    (model_dir / REWARD_NORMALIZER_FILE).unlink(missing_ok=True)
