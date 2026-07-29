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
            is_new_best_score = False
            bk2_file_path = Path(info["episode_bk2_path"])
            model_path = Path(self.config.paths.models)
            reward_path = (
                model_path
                / self.config.training.game
                / self.config.training.savestate
                / "rewards"
                / f"{info['episode_state']}.score"
            )
            curriculum_path = (
                model_path
                / self.config.training.game
                / self.config.training.savestate
                / "curriculum"
                / f"{info['episode_state']}.state"
            )
            score = int(float(info["episode"]["r"]))
            if reward_path.exists():
                with reward_path.open("r") as f:
                    best_score = int(float(f.read()))
                if score > best_score:
                    is_new_best_score = True
                    with reward_path.open("w") as f:
                        f.write(str(score))
            else:
                reward_path.parent.mkdir(parents=True, exist_ok=True)
                with reward_path.open("w") as f:
                    f.write(str(score))
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
            if is_new_best_score and not info["won"]:
                logger.debug("New best score: {:.3f}", score)
                files = {}

                with bk2_file_path.open("rb") as bk2_stream:
                    files["bk2_file"] = (bk2_file_path.name, bk2_stream, "application/zip")

                    if curriculum_path.exists():
                        with curriculum_path.open("rb") as curriculum_stream:
                            files["curriculum_file"] = (
                                curriculum_path.name,
                                curriculum_stream,
                                "application/octet-stream",
                            )
                            response = httpx.post(
                                f"{self.config.upload.url}/runs",
                                headers={"X-API-Key": self.config.upload.api_key},
                                data={
                                    "game": self.config.training.game,
                                    "category": self.config.training.savestate,
                                    "curriculum": info["episode_state"],
                                    "action_repeat": 4,
                                    "episode_number": info["episode_number"],
                                    "type": "TRAINING",
                                },
                                files=files,
                            )
                    else:
                        response = httpx.post(
                            f"{self.config.upload.url}/runs",
                            headers={"X-API-Key": self.config.upload.api_key},
                            data={
                                "game": self.config.training.game,
                                "category": self.config.training.savestate,
                                "curriculum": info["episode_state"],
                                "action_repeat": 4,
                                "episode_number": info["episode_number"],
                                "type": "TRAINING",
                            },
                            files=files,
                        )

                response.raise_for_status()
            if not info["won"] or not info["full_run"]:
                bk2_file_path.unlink()
                continue
            logger.info("Uploading winning episode {}", bk2_file_path.name)
            with bk2_file_path.open("rb") as stream:
                response = httpx.post(
                    f"{self.config.upload.url}/runs",
                    headers={"X-API-Key": self.config.upload.api_key},
                    data={
                        "game": self.config.training.game,
                        "category": self.config.training.savestate,
                        "type": "WON",
                        "action_repeat": 4,
                        "episode_number": info["episode_number"],
                    },
                    files={"bk2_file": (bk2_file_path.name, stream, "application/zip")},
                )
            response.raise_for_status()
            bk2_file_path.unlink()
            logger.success("Uploaded winning episode {}", bk2_file_path.name)
        return True
