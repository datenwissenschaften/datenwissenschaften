import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from box import Box

from datenwissenschaften.training.runner_stats import RunnerStatsPublisher


def _config(models: Path) -> Box:
    return Box(
        {
            "paths": {"models": models},
            "training": {
                "game": "Example-Nes",
                "savestate": "Level1",
                "fingerprint": "abc123",
                "runner_id": "runner-1",
                "runner_name": "Example runner",
            },
            "upload": {"url": "https://example.test", "api_key": "secret"},
        }
    )


def test_publishes_and_persists_completed_episode_statistics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    requests: list[dict[str, object]] = []

    def post(url: str, **kwargs: object) -> Mock:
        requests.append({"url": url, **kwargs})
        return Mock()

    monkeypatch.setattr("datenwissenschaften.training.runner_stats.httpx.post", post)
    monkeypatch.setattr("datenwissenschaften.training.runner_stats.time.monotonic", lambda: 100.0)
    publisher = RunnerStatsPublisher(_config(tmp_path / "models"))
    publisher.locals = {
        "dones": [True, True],
        "infos": [
            {
                "episode": {"r": -2.0, "l": 120},
                "episode_number": 41,
                "action_repeat": 3,
                "won": False,
            },
            {
                "episode": {"r": 8.0, "l": 60},
                "episode_number": 42,
                "action_repeat": 4,
                "won": True,
            },
        ],
    }

    assert publisher._on_step()

    payload = requests[0]["json"]
    assert payload["runner_id"] == "runner-1"
    assert payload["current_game"] == "Example-Nes"
    assert payload["wins"] == 1
    assert payload["episodes"] == 42
    assert payload["best_fitness"] == 8.0
    assert payload["average_training_seconds"] == pytest.approx(5.0)
    assert payload["latest_training_seconds"] == 4.0
    assert requests[0]["url"] == "https://example.test/runner/stats"
    persisted = json.loads(publisher.path.read_text(encoding="utf-8"))
    assert persisted["timed_episodes"] == 2
    assert RunnerStatsPublisher(_config(tmp_path / "models")).stats == persisted


def test_limits_pocketbase_updates_to_publish_interval(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    post = Mock(return_value=Mock())
    current_time = [100.0]
    monkeypatch.setattr("datenwissenschaften.training.runner_stats.httpx.post", post)
    monkeypatch.setattr("datenwissenschaften.training.runner_stats.time.monotonic", lambda: current_time[0])
    publisher = RunnerStatsPublisher(_config(tmp_path / "models"))
    publisher.locals = {
        "dones": [True],
        "infos": [{"episode": {"r": 1.0, "l": 60}, "episode_number": 1, "action_repeat": 1, "won": False}],
    }

    publisher._on_step()
    current_time[0] = 110.0
    publisher._on_step()

    post.assert_called_once()
