from pathlib import Path

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
