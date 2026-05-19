import json
import os

import requests
from itsdangerous import Signer
from stable_baselines3.common.callbacks import BaseCallback

from datenwissenschaften.console import ui_info
from datenwissenschaften.runtime import get_runtime


class UploadEpisodeCallback(BaseCallback):
    def __init__(self):
        super().__init__()

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        runtime = get_runtime()
        best_episode_path = runtime.get_state_value("best_episode")

        if not best_episode_path or not os.path.exists(best_episode_path):
            return True

        metadata = runtime.get_model_metadata(self.model)
        metadata["num_timesteps"] = self.num_timesteps
        metadata["game"] = runtime.game
        metadata["savestate"] = runtime.savestate

        secret_key = os.environ.get("RETRO_SPEEDLAB_API_KEY")
        if not secret_key:
            ui_info("RETRO_SPEEDLAB_API_KEY not set. Skipping episode upload.")
            return True

        model_id = os.environ.get("RETRO_SPEEDLAB_MODEL_ID")
        if not model_id:
            ui_info("RETRO_SPEEDLAB_MODEL_ID not set. Skipping episode upload.")
            return True

        metadata["model_id"] = model_id

        signer = Signer(secret_key)
        metadata_json = json.dumps(metadata, indent=4, sort_keys=True)
        signed_metadata = signer.sign(metadata_json.encode("utf-8"))

        upload_url = os.environ.get("RETRO_SPEEDLAB_UPLOAD_URL")

        try:
            with open(best_episode_path, "rb") as f:
                files = {
                    "bk2_file": (os.path.basename(best_episode_path), f, "application/octet-stream"),
                    "metadata_file": ("metadata.json.signed", signed_metadata, "application/octet-stream"),
                }
                data = {
                    "game_id": runtime.game,
                    "model_id": model_id,
                    "savestate": runtime.savestate,
                }
                headers = {
                    "X-API-Key": secret_key,
                }
                ui_info(f"Uploading episode to {upload_url}...")
                response = requests.post(upload_url, files=files, data=data, headers=headers, timeout=30)
                response.raise_for_status()
                ui_info("Episode uploaded successfully.")
        except Exception as e:
            ui_info(f"Failed to upload episode: {e}")

        return True
