from pathlib import Path

import httpx
from box import Box
from loguru import logger
from stable_baselines3.common.callbacks import BaseCallback


class WinningEpisodeUploader(BaseCallback):
    def __init__(self, config: Box) -> None:
        super().__init__()
        self.config = config

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"], strict=True):
            if not done:
                continue
            path = Path(info["episode_bk2_path"])
            episode = info["episode"]
            logger.debug(
                "Episode finished: reward={:.3f}, steps={}, start={}, end={}, full_run={}, won={}",
                episode["r"],
                episode["l"],
                info["episode_state"],
                info["state"],
                info["full_run"],
                info["won"],
            )
            if not info["won"] or not info["full_run"]:
                path.unlink()
                continue
            logger.info("Uploading winning episode {}", path.name)
            with path.open("rb") as stream:
                response = httpx.post(
                    f"{self.config.upload.url}/runs",
                    headers={"X-API-Key": self.config.upload.api_key},
                    data={"game": self.config.training.game, "category": self.config.training.savestate, "type": "WON"},
                    files={"bk2_file": (path.name, stream, "application/zip")},
                )
            response.raise_for_status()
            path.unlink()
            logger.success("Uploaded winning episode {}", path.name)
        return True
