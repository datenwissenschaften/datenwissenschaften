from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

from datenwissenschaften.training.trainer import CHECKPOINT_INTERVAL, _checkpoint_callback


def test_checkpoint_callback_saves_one_shared_agent(
    monkeypatch,
    tmp_path: Path,
) -> None:
    environment = Mock()
    model = SimpleNamespace(num_timesteps=CHECKPOINT_INTERVAL - 1, get_env=Mock(return_value=environment))
    save_model = Mock()
    save_normalizer = Mock()
    monkeypatch.setattr(
        "datenwissenschaften.training.trainer.atomic_save",
        save_model,
    )
    monkeypatch.setattr(
        "datenwissenschaften.training.trainer.save_reward_normalizer",
        save_normalizer,
    )
    checkpoint = tmp_path / "model"
    callback = _checkpoint_callback(model, checkpoint)

    assert callback({"self": model}, {})
    save_model.assert_not_called()
    save_normalizer.assert_not_called()

    model.num_timesteps = CHECKPOINT_INTERVAL
    assert callback({"self": model}, {})
    save_model.assert_called_once_with(model, checkpoint)
    save_normalizer.assert_has_calls([call(environment, tmp_path)])
