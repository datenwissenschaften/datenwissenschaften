from pathlib import Path

from box import Box

from datenwissenschaften.models.path import model_directory


def test_model_directory_includes_fingerprint_after_savestate() -> None:
    config = Box(
        {
            "paths": {"models": Path("/mnt/fastdata/models")},
            "training": {
                "game": "Example-Nes",
                "savestate": "Level1",
                "fingerprint": "abc123",
            },
        }
    )

    assert model_directory(config) == Path("/mnt/fastdata/models/Example-Nes/Level1/abc123/agents")
