from pathlib import Path
from unittest.mock import Mock

import pytest

from datenwissenschaften.retro.environment import _create_environment


def test_creates_visible_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    retro_environment = Mock()
    make = Mock(return_value=retro_environment)
    wrapper = Mock()
    monkeypatch.setattr("datenwissenschaften.retro.environment.stable_retro.make", make)

    _create_environment(wrapper, "Example-Nes-v0", "Level1", "human", tmp_path, 0)

    make.assert_called_once_with(
        "Example-Nes-v0",
        "Level1",
        render_mode="human",
        record=tmp_path / "episodes" / "0",
    )
    wrapper.assert_called_once_with(retro_environment, model_dir=tmp_path)
