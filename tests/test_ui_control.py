from pathlib import Path

from datenwissenschaften.ui.control import ModelResetRequest, perform_model_reset


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
