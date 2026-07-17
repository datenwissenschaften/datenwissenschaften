from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from datenwissenschaften.callbacks.upload_episode_callback import UploadEpisodeCallback
from datenwissenschaften.settings import UploadSettings


def test_successful_episode_flushes_immediately(monkeypatch):
    callback = UploadEpisodeCallback(UploadSettings(url="https://example.test", api_key="key"))
    callback.locals = {
        "rewards": np.asarray([1.0]),
        "dones": np.asarray([True]),
        "infos": [{"won": True, "started_from_initial_savestate": True}],
    }
    monkeypatch.setattr(callback, "_finish_episode", lambda _index, episode: setattr(episode, "won", True))
    flush = Mock()
    monkeypatch.setattr(callback, "_flush_successful_episodes", flush)

    callback._on_step()

    flush.assert_called_once_with()


def test_episode_path_resolves_nested_runner_recording_layout(tmp_path: Path):
    filename = "Game-Level1-000003.bk2"
    nested = tmp_path / "Game" / "Level1" / "2" / filename
    nested.parent.mkdir(parents=True)
    nested.write_bytes(b"movie")
    runtime = SimpleNamespace(record_dir=tmp_path, game="Game", savestate="Level1")

    resolved = UploadEpisodeCallback._resolve_episode_path(str(tmp_path / "2" / filename), runtime)

    assert resolved == str(nested)
