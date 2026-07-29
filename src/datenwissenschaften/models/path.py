from pathlib import Path

from box import Box


def model_directory(config: Box) -> Path:
    return config.paths.models / config.training.game / config.training.savestate / config.training.fingerprint
