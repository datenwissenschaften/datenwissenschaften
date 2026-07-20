from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from datenwissenschaften.callbacks.episode_record import EpisodeRecord
from datenwissenschaften.callbacks.upload_episode_callback import UploadEpisodeCallback
from datenwissenschaften.settings import UploadSettings
from loguru import logger


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


def test_finish_episode_preserves_environment_reported_movie_path():
    callback = UploadEpisodeCallback(UploadSettings(url="https://example.test", api_key="key"))
    callback._ensure_episode_slots(1)
    episode = EpisodeRecord(0, 0)
    episode.bk2_path = "actual-0042.bk2"

    callback._finish_episode(0, episode)

    assert callback.completed_episodes[0].bk2_path == "actual-0042.bk2"


def test_checkpoint_win_explains_why_it_is_not_uploaded(monkeypatch):
    callback = UploadEpisodeCallback(UploadSettings(url="https://example.test", api_key="key"))
    checkpoint_won = EpisodeRecord(0, 0)
    checkpoint_won.won = True
    checkpoint_won.started_from_initial_savestate = False
    callback.completed_episodes = [checkpoint_won]
    messages = []
    sink = logger.add(lambda message: messages.append(str(message)), format="{message}")
    monkeypatch.setattr(callback, "_flush_successful_episodes", lambda: None)

    try:
        callback._on_rollout_end()
    finally:
        logger.remove(sink)

    assert any("only accepts complete runs" in message for message in messages)
