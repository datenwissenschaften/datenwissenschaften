from pathlib import Path

import cv2
import numpy as np
from datenwissenschaften.ui import server
from datenwissenschaften.ui.control import ModelResetRequest, perform_model_reset
from datenwissenschaften.ui.server import generated_source, generated_sources


def test_generated_sources_only_exposes_runner_project_files(tmp_path: Path):
    (tmp_path / "runner.py").write_text("print('runner')\n", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("do not expose", encoding="utf-8")

    assert generated_sources(tmp_path) == [{"path": "runner.py", "language": "python", "size": 16}]


def test_generated_config_source_redacts_upload_api_key(tmp_path: Path):
    (tmp_path / "config.yaml").write_text(
        'training:\n  game: "Test"\nupload:\n  url: "https://example.test"\n  api_key: "secret"\nui:\n  enable: true\n',
        encoding="utf-8",
    )

    source = generated_source("config.yaml", tmp_path)

    assert 'api_key: "[REDACTED]"' in source["content"]
    assert 'api_key: "secret"' not in source["content"]


def test_learned_enemy_gallery_only_lists_cached_png_files(tmp_path: Path, monkeypatch):
    root = tmp_path / "learned_enemies" / "Game" / "Level1" / "Explore"
    root.mkdir(parents=True)
    sprite = np.zeros((8, 8, 4), dtype=np.uint8)
    sprite[2:6, 2:6] = (20, 80, 255, 255)
    cv2.imwrite(str(root / "enemy.png"), sprite)
    cv2.imwrite(str(root / "old-background.png"), np.zeros((8, 8, 3), dtype=np.uint8))
    (root / "ignored.txt").write_text("not an image", encoding="utf-8")
    monkeypatch.setattr(server, "get_runtime", lambda: type("Runtime", (), {"cache_dir": tmp_path})())

    enemies = server.learned_enemies()

    assert enemies == [
        {
            "id": "enemy",
            "path": "Game/Level1/Explore/enemy.png",
            "game": "Game",
            "savestate": "Level1",
            "state": "Explore",
            "size": (root / "enemy.png").stat().st_size,
        }
    ]
    assert not (root / "old-background.png").exists()
    assert server.learned_enemy_path(enemies[0]["path"]) == root / "enemy.png"


def test_model_reset_deletes_and_recreates_all_runner_artifact_directories(tmp_path: Path):
    artifact_dirs = tuple(tmp_path / name for name in ("models", "recordings", "cache"))
    for path in artifact_dirs:
        path.mkdir()
        (path / "old-artifact").write_text("old", encoding="utf-8")
    unrelated = tmp_path / "roms"
    unrelated.mkdir()
    (unrelated / "game.rom").write_text("keep", encoding="utf-8")

    perform_model_reset(
        ModelResetRequest(
            game="TestGame",
            model_dir=artifact_dirs[0],
            artifact_dirs=artifact_dirs,
        )
    )

    assert all(path.is_dir() and not list(path.iterdir()) for path in artifact_dirs)
    assert (unrelated / "game.rom").read_text(encoding="utf-8") == "keep"
