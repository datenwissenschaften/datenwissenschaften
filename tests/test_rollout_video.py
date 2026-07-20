import json
from pathlib import Path
from types import SimpleNamespace

from datenwissenschaften import rollout_video, rollout_video_playback
from datenwissenschaften.callbacks.episode_record import EpisodeRecord


def _episode(
    path: Path,
    curriculum: str,
    score: float,
    steps: int = 10,
    *,
    curriculum_succeeded: bool = False,
    env_index: int = 0,
    episode_index: int = 0,
) -> EpisodeRecord:
    episode = EpisodeRecord(env_index, episode_index)
    episode.bk2_path = str(path)
    episode.curriculum_state = curriculum
    episode.score = score
    episode.step_count = steps
    episode.curriculum_succeeded = curriculum_succeeded
    return episode


def test_records_highest_scoring_episode_per_curriculum(monkeypatch, tmp_path: Path):
    worker_dir = tmp_path / "Game" / "Level1" / "0"
    worker_dir.mkdir(parents=True)
    low_path = worker_dir / "low.bk2"
    high_path = worker_dir / "high.bk2"
    other_path = worker_dir / "other.bk2"
    for path in (low_path, high_path, other_path):
        path.write_bytes(b"movie")
    runtime = SimpleNamespace(
        record_dir=tmp_path,
        savestate="Level1",
        game="Game",
        paths=SimpleNamespace(roms_path=tmp_path / "roms"),
    )
    monkeypatch.setattr(rollout_video, "get_runtime", lambda: runtime)

    rendered = []

    def render(command, **_kwargs):
        assert command[2] == "datenwissenschaften.rollout_video_playback"
        assert command[3:5] == ["--roms-dir", str(tmp_path / "roms")]
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
    assert metadata["environment"] == 0
    assert metadata["recording"] == "high.bk2"


def test_stage_completing_episode_is_preferred_over_higher_failed_score(monkeypatch, tmp_path: Path):
    worker_dir = tmp_path / "Game" / "Level1" / "0"
    worker_dir.mkdir(parents=True)
    failed_path = worker_dir / "failed.bk2"
    completed_path = worker_dir / "completed.bk2"
    failed_path.write_bytes(b"failed")
    completed_path.write_bytes(b"completed")
    runtime = SimpleNamespace(
        record_dir=tmp_path,
        savestate="Level1",
        game="Game",
        paths=SimpleNamespace(roms_path=tmp_path / "roms"),
    )
    monkeypatch.setattr(rollout_video, "get_runtime", lambda: runtime)

    def render(command, **_kwargs):
        Path(command[-1]).with_suffix(".mp4").write_bytes(b"video")

    monkeypatch.setattr(rollout_video.subprocess, "run", render)

    videos = rollout_video.record_rollout_videos(
        [
            _episode(failed_path, "FindDispenser", 10_000.0),
            _episode(completed_path, "FindDispenser", 100.0, curriculum_succeeded=True),
        ],
        rollout=4,
    )

    assert videos == [completed_path.with_suffix(".mp4")]
    metadata = json.loads(completed_path.with_suffix(".rollout.json").read_text())
    assert metadata["curriculum_succeeded"] is True


def test_playback_imports_configured_roms_first(monkeypatch):
    calls = []
    monkeypatch.setattr(rollout_video_playback, "import_roms", lambda path: calls.append(("roms", path)))
    monkeypatch.setattr(
        rollout_video_playback.retro,
        "make",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    def playback(movie, args, monitor_csv):
        calls.append((movie, args.no_audio, monitor_csv))
        rollout_video_playback.retro.make("Game")

    monkeypatch.setattr(rollout_video_playback, "play_movie", playback)

    rollout_video_playback.main(["--roms-dir", "/roms", "--no-audio", "episode.bk2"])

    assert calls == [
        ("roms", "/roms"),
        ("episode.bk2", True, None),
        (("Game",), {"render_mode": "rgb_array"}),
    ]


def test_episode_score_uses_terminal_monitor_score():
    episode = EpisodeRecord(0, 0)
    episode.add_step({"won": False, "curriculum_state": "Explore", "extrinsic_reward": 2.0}, 99.0)
    episode.add_step({"won": False, "episode": {"r": 12.5}}, 3.0)

    assert episode.curriculum_state == "Explore"
    assert episode.score == 12.5


def test_episode_records_curriculum_success():
    episode = EpisodeRecord(0, 0)

    episode.add_step({"won": False, "curriculum_state": "FindDispenser"}, 1.0)
    episode.add_step({"won": False, "curriculum_succeeded": True}, 1.0)

    assert episode.curriculum_succeeded is True
    assert episode.clone().curriculum_succeeded is True


def test_episode_records_curriculum_mastery():
    episode = EpisodeRecord(0, 0)

    episode.add_step(
        {
            "won": False,
            "curriculum_state": "FindDispenser",
            "curriculum_succeeded": True,
            "curriculum_mastered": True,
        },
        1.0,
    )

    assert episode.curriculum_mastered is True
    assert episode.clone().curriculum_mastered is True


def test_recording_resolution_never_crosses_worker_directories(tmp_path: Path):
    filename = "Game-Level1-000007.bk2"
    requested = tmp_path / "7" / filename
    correct = tmp_path / "Game" / "Level1" / "7" / filename
    wrong = tmp_path / "Game" / "Level1" / "3" / filename
    correct.parent.mkdir(parents=True)
    wrong.parent.mkdir(parents=True)
    correct.write_bytes(b"worker 7")
    wrong.write_bytes(b"worker 3")

    assert (
        rollout_video._resolve_recording(
            str(requested),
            tmp_path,
            game="Game",
            savestate="Level1",
            env_index=7,
        )
        == correct.resolve()
    )


def test_existing_recording_from_another_worker_is_rejected(tmp_path: Path):
    filename = "Game-Level1-000007.bk2"
    wrong = tmp_path / "Game" / "Level1" / "3" / filename
    wrong.parent.mkdir(parents=True)
    wrong.write_bytes(b"worker 3")

    assert (
        rollout_video._resolve_recording(
            str(wrong),
            tmp_path,
            game="Game",
            savestate="Level1",
            env_index=7,
        )
        is None
    )


def test_parallel_workers_with_same_filename_render_the_selected_worker(monkeypatch, tmp_path: Path):
    filename = "Game-Level1-000007.bk2"
    worker_2 = tmp_path / "Game" / "Level1" / "2" / filename
    worker_9 = tmp_path / "Game" / "Level1" / "9" / filename
    worker_2.parent.mkdir(parents=True)
    worker_9.parent.mkdir(parents=True)
    worker_2.write_bytes(b"worker 2")
    worker_9.write_bytes(b"worker 9")
    runtime = SimpleNamespace(
        record_dir=tmp_path,
        savestate="Level1",
        game="Game",
        paths=SimpleNamespace(roms_path=tmp_path / "roms"),
    )
    monkeypatch.setattr(rollout_video, "get_runtime", lambda: runtime)
    rendered = []

    def render(command, **_kwargs):
        source = Path(command[-1])
        rendered.append(source)
        source.with_suffix(".mp4").write_bytes(b"video")

    monkeypatch.setattr(rollout_video.subprocess, "run", render)

    videos = rollout_video.record_rollout_videos(
        [
            _episode(worker_2, "Explore", 4.0, env_index=2, episode_index=7),
            _episode(worker_9, "Explore", 12.0, env_index=9, episode_index=7),
        ],
        rollout=3,
    )

    assert rendered == [worker_9.resolve()]
    assert videos == [worker_9.with_suffix(".mp4").resolve()]
    assert worker_2.read_bytes() == b"worker 2"


def test_stale_video_is_not_accepted_as_fresh_output(monkeypatch, tmp_path: Path):
    source = tmp_path / "Game" / "Level1" / "4" / "Game-Level1-000002.bk2"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"worker 4")
    stale_video = source.with_suffix(".mp4")
    stale_metadata = source.with_suffix(".rollout.json")
    stale_video.write_bytes(b"stale video")
    stale_metadata.write_text("stale metadata", encoding="utf-8")
    runtime = SimpleNamespace(
        record_dir=tmp_path,
        savestate="Level1",
        game="Game",
        paths=SimpleNamespace(roms_path=tmp_path / "roms"),
    )
    monkeypatch.setattr(rollout_video, "get_runtime", lambda: runtime)
    monkeypatch.setattr(rollout_video.subprocess, "run", lambda *_args, **_kwargs: None)

    videos = rollout_video.record_rollout_videos(
        [_episode(source, "Explore", 9.0, env_index=4, episode_index=2)],
        rollout=7,
    )

    assert videos == []
    assert not stale_video.exists()
    assert not stale_metadata.exists()
