import os
import tempfile

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


class SaveModelCallback(BaseCallback):
    def __init__(self):
        super().__init__()

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        runtime = get_runtime()
        model_path = runtime.get_model_path(runtime.game)
        atomic_save(self.model, model_path)

        ui_model_saved(steps=self.num_timesteps, model_path=f"{model_path}.zip")
        return True
