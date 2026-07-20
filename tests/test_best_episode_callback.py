from datenwissenschaften.callbacks import best_episode_callback
from datenwissenschaften.callbacks.best_episode_callback import BestEpisodeCallback
from datenwissenschaften.callbacks.episode_record import EpisodeRecord


def test_mastering_episode_is_recorded_immediately(monkeypatch):
    callback = BestEpisodeCallback()
    callback._ensure_episode_slots(1)
    episode = EpisodeRecord(0, 3)
    episode.curriculum_state = "FindDispenser"
    episode.curriculum_succeeded = True
    episode.curriculum_mastered = True
    recorded = []

    monkeypatch.setattr(callback, "_bk2_path", lambda env_index, episode_index: "mastered.bk2")
    monkeypatch.setattr(
        best_episode_callback,
        "record_rollout_videos",
        lambda episodes, rollout: recorded.append((episodes, rollout)) or ["mastered.mp4"],
    )

    callback._finish_episode(0, episode)

    assert recorded == [([episode.clone()], 1)]
    assert callback.episodes == []


def test_failed_immediate_recording_is_retried_at_rollout_end(monkeypatch):
    callback = BestEpisodeCallback()
    callback._ensure_episode_slots(1)
    episode = EpisodeRecord(0, 3)
    episode.curriculum_mastered = True
    calls = []

    monkeypatch.setattr(callback, "_bk2_path", lambda env_index, episode_index: "mastered.bk2")
    monkeypatch.setattr(
        best_episode_callback,
        "record_rollout_videos",
        lambda episodes, rollout: calls.append((episodes, rollout)) or [],
    )

    callback._finish_episode(0, episode)
    callback._on_rollout_end()

    assert [rollout for _, rollout in calls] == [1, 1]
    assert callback.episodes == []


def test_finish_episode_preserves_environment_reported_movie_path(monkeypatch):
    callback = BestEpisodeCallback()
    callback._ensure_episode_slots(1)
    episode = EpisodeRecord(0, 0)
    episode.bk2_path = "actual-0042.bk2"
    inferred = []
    monkeypatch.setattr(callback, "_bk2_path", lambda *args: inferred.append(args) or "wrong.bk2")

    callback._finish_episode(0, episode)

    assert callback.episodes[0].bk2_path == "actual-0042.bk2"
    assert inferred == []
