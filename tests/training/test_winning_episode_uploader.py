from pathlib import Path
from unittest.mock import Mock

import pytest
from box import Box

from datenwissenschaften.training.winning_episode_uploader import WinningEpisodeUploader


def test_winning_full_run_uploads_required_metadata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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
                "episode_state": "Start",
                "full_run": True,
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
        "curriculum": "Start",
        "type": "WON",
        "action_repeat": 4,
        "episode_number": 42,
    }
    assert not recording.exists()
