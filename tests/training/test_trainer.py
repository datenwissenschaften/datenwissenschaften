from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from datenwissenschaften.training.trainer import CHECKPOINT_INTERVAL, _checkpoint_callback


def test_checkpoint_callback_saves_one_shared_agent(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model = SimpleNamespace(num_timesteps=CHECKPOINT_INTERVAL - 1)
    save_model = Mock()
    monkeypatch.setattr(
        "datenwissenschaften.training.trainer.atomic_save",
        save_model,
    )
    checkpoint = tmp_path / "model"
    callback = _checkpoint_callback(model, checkpoint)

    assert callback({"self": model}, {})
    save_model.assert_not_called()

    model.num_timesteps = CHECKPOINT_INTERVAL
    assert callback({"self": model}, {})
    save_model.assert_called_once_with(model, checkpoint)
