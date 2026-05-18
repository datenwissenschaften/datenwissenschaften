import json
import os
import tempfile

from itsdangerous import Signer
from stable_baselines3.common.callbacks import BaseCallback

from datenwissenschaften.console import ui_model_saved
from datenwissenschaften.runtime import get_runtime


def atomic_save(model, model_path: str) -> None:
    target = model_path + ".zip"
    dir_name = os.path.dirname(target)
    fd, tmp_path = tempfile.mkstemp(suffix=".tmp", dir=dir_name)
    os.close(fd)
    os.remove(tmp_path)
    tmp_base = tmp_path.removesuffix(".tmp")
    tmp_zip = tmp_base + ".zip"
    try:
        model.save(tmp_base)
        os.replace(tmp_zip, target)
    except Exception:
        if os.path.exists(tmp_zip):
            os.remove(tmp_zip)
        raise


def atomic_write(path: str, data: bytes) -> None:
    dir_name = os.path.dirname(path)
    fd, tmp_path = tempfile.mkstemp(suffix=".tmp", dir=dir_name)
    os.close(fd)
    try:
        with open(tmp_path, "wb") as f:
            f.write(data)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


class SaveModelCallback(BaseCallback):
    def __init__(self):
        super().__init__()

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        runtime = get_runtime()
        model_path = runtime.get_model_path(runtime.game)
        atomic_save(self.model, model_path)

        metadata = {
            "action_space": str(self.model.action_space),
            "batch_size": getattr(self.model, "batch_size", None),
            "clip_range": str(getattr(self.model, "clip_range", None)),
            "clip_range_vf": str(getattr(self.model, "clip_range_vf", None)),
            "device": str(getattr(self.model, "device", None)),
            "ent_coef": getattr(self.model, "ent_coef", None),
            "gae_lambda": getattr(self.model, "gae_lambda", None),
            "gamma": getattr(self.model, "gamma", None),
            "lr_schedule": str(getattr(self.model, "lr_schedule", None)),
            "max_grad_norm": getattr(self.model, "max_grad_norm", None),
            "n_envs": getattr(self.model, "n_envs", None),
            "n_epochs": getattr(self.model, "n_epochs", None),
            "n_steps": getattr(self.model, "n_steps", None),
            "normalize_advantage": getattr(self.model, "normalize_advantage", None),
            "num_timesteps": self.num_timesteps,
            "observation_space": str(self.model.observation_space),
            "policy_class": str(getattr(self.model, "policy_class", None)),
            "policy_kwargs": getattr(self.model, "policy_kwargs", None),
            "sde_sample_freq": getattr(self.model, "sde_sample_freq", None),
            "seed": getattr(self.model, "seed", None),
            "target_kl": getattr(self.model, "target_kl", None),
            "use_sde": getattr(self.model, "use_sde", None),
            "verbose": getattr(self.model, "verbose", None),
            "vf_coef": getattr(self.model, "vf_coef", None),
            "_total_timesteps": getattr(self.model, "_total_timesteps", None),
        }

        secret_key = os.environ.get("RETRO_ARENA_API_KEY")
        signer = Signer(secret_key)
        metadata_json = json.dumps(metadata, indent=4, sort_keys=True)
        signed_metadata = signer.sign(metadata_json.encode("utf-8"))

        metadata_path = f"{model_path}.json"
        atomic_write(metadata_path, signed_metadata)

        ui_model_saved(steps=self.num_timesteps, model_path=f"{model_path}.zip")
