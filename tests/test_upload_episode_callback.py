from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from datenwissenschaften.callbacks.episode_record import EpisodeRecord
from datenwissenschaften.callbacks.upload_episode_callback import UploadEpisodeCallback
from datenwissenschaften.settings import UploadSettings


def test_uploads_every_won_episode_started_from_initial_savestate(monkeypatch):
    callback = UploadEpisodeCallback(UploadSettings(url="https://example.test", api_key="key"))
    flush = Mock()
    monkeypatch.setattr(callback, "_flush_successful_episodes", flush)
    won = EpisodeRecord(0, 0)
    won.won = True
    won.started_from_initial_savestate = True
    won.score = 10.0
    second_won = EpisodeRecord(1, 0)
    second_won.won = True
    second_won.started_from_initial_savestate = True
    checkpoint_won = EpisodeRecord(2, 0)
    checkpoint_won.won = True
    checkpoint_won.started_from_initial_savestate = False
    lost = EpisodeRecord(3, 0)
    lost.started_from_initial_savestate = True
    lost.score = 20.0
    callback.completed_episodes = [won, second_won, checkpoint_won, lost]

    callback._on_rollout_end()

    flush.assert_called_once_with()
    assert len(callback.successful_episodes) == 2
    assert all(episode.won for episode in callback.successful_episodes)
    assert all(episode.started_from_initial_savestate is True for episode in callback.successful_episodes)


def test_episode_path_resolves_nested_runner_recording_layout(tmp_path: Path):
    filename = "Game-Level1-000003.bk2"
    nested = tmp_path / "Game" / "Level1" / "2" / filename
    nested.parent.mkdir(parents=True)
    nested.write_bytes(b"movie")
    runtime = SimpleNamespace(record_dir=tmp_path, game="Game", savestate="Level1")

    resolved = UploadEpisodeCallback._resolve_episode_path(str(tmp_path / "2" / filename), runtime)

    assert resolved == str(nested)
