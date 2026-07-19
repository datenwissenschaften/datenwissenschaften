import json
from pathlib import Path
from types import SimpleNamespace

from datenwissenschaften import rollout_video, rollout_video_playback
from datenwissenschaften.callbacks.episode_record import EpisodeRecord


def _episode(path: Path, curriculum: str, score: float, steps: int = 10) -> EpisodeRecord:
    episode = EpisodeRecord(0, 0)
    episode.bk2_path = str(path)
    episode.curriculum_state = curriculum
    episode.score = score
    episode.step_count = steps
    return episode


def test_records_highest_scoring_episode_per_curriculum(monkeypatch, tmp_path: Path):
    low_path = tmp_path / "low.bk2"
    high_path = tmp_path / "high.bk2"
    other_path = tmp_path / "other.bk2"
    for path in (low_path, high_path, other_path):
        path.write_bytes(b"movie")
    runtime = SimpleNamespace(record_dir=tmp_path, savestate="Level1", game="Game")
    monkeypatch.setattr(rollout_video, "get_runtime", lambda: runtime)

    rendered = []

    def render(command, **_kwargs):
        assert command[2] == "datenwissenschaften.rollout_video_playback"
        source = Path(command[-1])
        source.with_suffix(".mp4").write_bytes(b"video")
        rendered.append(source.name)

    monkeypatch.setattr(rollout_video.subprocess, "run", render)

    videos = rollout_video.record_rollout_videos(
        [
            _episode(low_path, "Explore", 2.0),
            _episode(high_path, "Explore", 9.0),
            _episode(other_path, "Fight", 4.0),
        ],
        rollout=7,
    )

    assert rendered == ["high.bk2", "other.bk2"]
    assert videos == [high_path.with_suffix(".mp4"), other_path.with_suffix(".mp4")]
    metadata = json.loads(high_path.with_suffix(".rollout.json").read_text())
    assert metadata["curriculum"] == "Explore"
    assert metadata["rollout"] == 7
    assert metadata["score"] == 9.0


def test_playback_imports_configured_roms_first(monkeypatch):
    calls = []
    monkeypatch.setattr(rollout_video_playback, "import_roms", lambda: calls.append("roms"))
    monkeypatch.setattr(rollout_video_playback, "playback_main", lambda argv: calls.append(argv))

    rollout_video_playback.main(["--no-audio", "episode.bk2"])

    assert calls == ["roms", ["--no-audio", "episode.bk2"]]


def test_episode_score_uses_terminal_monitor_score():
    episode = EpisodeRecord(0, 0)
    episode.add_step({"won": False, "curriculum_state": "Explore", "extrinsic_reward": 2.0}, 99.0)
    episode.add_step({"won": False, "episode": {"r": 12.5}}, 3.0)

    assert episode.curriculum_state == "Explore"
    assert episode.score == 12.5
