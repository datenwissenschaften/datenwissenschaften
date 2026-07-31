from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest
from box import Box

from datenwissenschaften.training.winning_episode_uploader import WinningEpisodeUploader


def test_winning_run_uploads_required_metadata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    recording = tmp_path / "winning.bk2"
    recording.write_bytes(b"recording")
    requests: list[dict[str, object]] = []

    def post(url: str, **kwargs: object) -> Mock:
        requests.append({"url": url, **kwargs})
        return Mock()

    monkeypatch.setattr("datenwissenschaften.training.winning_episode_uploader.httpx.post", post)
    uploader = WinningEpisodeUploader(
        Box(
            {
                "paths": {"models": tmp_path / "models"},
                "training": {
                    "game": "Example-Nes",
                    "savestate": "Level1",
                    "fingerprint": "abc123",
                },
                "upload": {"url": "https://example.test", "api_key": "secret"},
            }
        )
    )
    uploader.locals = {
        "dones": [True],
        "infos": [
            {
                "episode": {"r": 100.0, "l": 500},
                "episode_bk2_path": str(recording),
                "episode_number": 42,
                "action_repeat": 3,
                "state": "Finish",
                "won": True,
            }
        ],
    }

    assert uploader._on_step()
    assert len(requests) == 1
    assert requests[0]["data"] == {
        "game": "Example-Nes",
        "category": "Level1",
        "type": "WON",
        "action_repeat": 3,
        "episode_number": 42,
    }
    assert not recording.exists()


def test_non_winning_score_is_not_uploaded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    recording = tmp_path / "first.bk2"
    recording.write_bytes(b"recording")
    requests: list[dict[str, object]] = []

    def post(url: str, **kwargs: object) -> Mock:
        requests.append({"url": url, **kwargs})
        return Mock()

    monkeypatch.setattr("datenwissenschaften.training.winning_episode_uploader.httpx.post", post)
    uploader = WinningEpisodeUploader(
        Box(
            {
                "paths": {"models": tmp_path / "models"},
                "training": {
                    "game": "Example-Nes",
                    "savestate": "Level1",
                    "fingerprint": "abc123",
                },
                "upload": {"url": "https://example.test", "api_key": "secret"},
            }
        )
    )
    uploader.locals = {
        "dones": [True],
        "infos": [
            {
                "episode": {"r": 10.0, "l": 50},
                "episode_bk2_path": str(recording),
                "episode_number": 1,
                "action_repeat": 2,
                "state": "Start",
                "won": False,
            }
        ],
    }

    assert uploader._on_step()
    assert not requests
    assert not recording.exists()
    assert not next((tmp_path / "models").rglob("best.score"), None)


@pytest.mark.parametrize("score", [100.0, 99.0])
def test_winning_score_must_improve_best(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    score: float,
) -> None:
    recording = tmp_path / "winning.bk2"
    recording.write_bytes(b"recording")
    requests: list[dict[str, object]] = []

    def post(url: str, **kwargs: object) -> Mock:
        requests.append({"url": url, **kwargs})
        return Mock()

    monkeypatch.setattr("datenwissenschaften.training.winning_episode_uploader.httpx.post", post)
    best_score = tmp_path / "models" / "Example-Nes" / "Level1" / "abc123" / "agents" / "best.score"
    best_score.parent.mkdir(parents=True)
    best_score.write_text("100.0", encoding="utf-8")
    uploader = WinningEpisodeUploader(
        Box(
            {
                "paths": {"models": tmp_path / "models"},
                "training": {
                    "game": "Example-Nes",
                    "savestate": "Level1",
                    "fingerprint": "abc123",
                },
                "upload": {"url": "https://example.test", "api_key": "secret"},
            }
        )
    )
    uploader.locals = {
        "dones": [True],
        "infos": [
            {
                "episode": {"r": score, "l": 50},
                "episode_bk2_path": str(recording),
                "episode_number": 1,
                "action_repeat": 4,
                "state": "Finish",
                "won": True,
            }
        ],
    }

    assert uploader._on_step()
    assert not requests
    assert not recording.exists()
    assert best_score.read_text(encoding="utf-8") == "100.0"


def test_better_winning_score_is_uploaded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    recording = tmp_path / "better.bk2"
    recording.write_bytes(b"recording")
    requests: list[dict[str, object]] = []

    def post(url: str, **kwargs: object) -> Mock:
        requests.append({"url": url, **kwargs})
        return Mock()

    monkeypatch.setattr("datenwissenschaften.training.winning_episode_uploader.httpx.post", post)
    best_score = tmp_path / "models" / "Example-Nes" / "Level1" / "abc123" / "agents" / "best.score"
    best_score.parent.mkdir(parents=True)
    best_score.write_text("100.0", encoding="utf-8")
    uploader = WinningEpisodeUploader(
        Box(
            {
                "paths": {"models": tmp_path / "models"},
                "training": {
                    "game": "Example-Nes",
                    "savestate": "Level1",
                    "fingerprint": "abc123",
                },
                "upload": {"url": "https://example.test", "api_key": "secret"},
            }
        )
    )
    uploader.locals = {
        "dones": [True],
        "infos": [
            {
                "episode": {"r": 101.0, "l": 50},
                "episode_bk2_path": str(recording),
                "episode_number": 1,
                "action_repeat": 4,
                "state": "Finish",
                "won": True,
            }
        ],
    }

    assert uploader._on_step()
    assert len(requests) == 1
    assert requests[0]["data"]["type"] == "WON"
    assert not recording.exists()
    assert best_score.read_text(encoding="utf-8") == "101.0"


def test_upload_failure_does_not_stop_training(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    recording = tmp_path / "failed-upload.bk2"
    recording.write_bytes(b"recording")

    def post(url: str, **kwargs: object) -> httpx.Response:
        del kwargs
        return httpx.Response(
            422,
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr("datenwissenschaften.training.winning_episode_uploader.httpx.post", post)
    uploader = WinningEpisodeUploader(
        Box(
            {
                "paths": {"models": tmp_path / "models"},
                "training": {
                    "game": "Example-Nes",
                    "savestate": "Level1",
                    "fingerprint": "abc123",
                },
                "upload": {"url": "https://example.test", "api_key": "secret"},
            }
        )
    )
    uploader.locals = {
        "dones": [True],
        "infos": [
            {
                "episode": {"r": 1.5, "l": 50},
                "episode_bk2_path": str(recording),
                "episode_number": 1,
                "action_repeat": 4,
                "state": "Finish",
                "won": True,
            }
        ],
    }

    assert uploader._on_step()
    assert recording.exists()
