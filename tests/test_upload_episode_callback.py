from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from datenwissenschaften.callbacks.episode_record import EpisodeRecord
from datenwissenschaften.callbacks.upload_episode_callback import UploadEpisodeCallback
from datenwissenschaften.settings import UploadSettings


def test_only_uploads_rollout_best_when_it_won(monkeypatch):
    callback = UploadEpisodeCallback(UploadSettings(url="https://example.test", api_key="key"))
    flush = Mock()
    monkeypatch.setattr(callback, "_flush_successful_episodes", flush)
    won = EpisodeRecord(0, 0)
    won.won = True
    won.score = 10.0
    lost = EpisodeRecord(1, 0)
    lost.score = 20.0
    callback.completed_episodes = [won, lost]

    callback._on_rollout_end()

    flush.assert_called_once_with()
    assert callback.successful_episodes == []

    won.score = 30.0
    callback.completed_episodes = [won, lost]
    callback._on_rollout_end()

    assert len(callback.successful_episodes) == 1
    assert callback.successful_episodes[0].won is True

    callback.successful_episodes.clear()
    won.curriculum_state = "ActivateScale"
    won.score = 10.0
    lost.curriculum_state = "Explore"
    callback.completed_episodes = [won, lost]
    callback._on_rollout_end()

    assert len(callback.successful_episodes) == 1
    assert callback.successful_episodes[0].curriculum_state == "ActivateScale"


def test_episode_path_resolves_nested_runner_recording_layout(tmp_path: Path):
    filename = "Game-Level1-000003.bk2"
    nested = tmp_path / "Game" / "Level1" / "2" / filename
    nested.parent.mkdir(parents=True)
    nested.write_bytes(b"movie")
    runtime = SimpleNamespace(record_dir=tmp_path, game="Game", savestate="Level1")

    resolved = UploadEpisodeCallback._resolve_episode_path(str(tmp_path / "2" / filename), runtime)

    assert resolved == str(nested)
