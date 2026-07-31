from pathlib import Path

import pytest

from datenwissenschaften.configuration.loader import load_config


def test_rejects_removed_environment_count(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        """
paths:
  roms: roms
  models: models
training:
  game: Example-Nes
  savestate: Level1
  num_envs: 4
  fingerprint: abc123
  runner_id: runner-1
  runner_name: Example runner
upload:
  url: https://example.test
  api_key: secret
log_level: INFO
""",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="Unsupported configuration value: training.num_envs"):
        load_config(config)
