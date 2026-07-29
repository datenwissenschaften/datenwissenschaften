from pathlib import Path

from datenwissenschaften.curriculum.stagnation import ScoreStagnation


def test_deletes_savestate_only_after_configured_patience(tmp_path: Path) -> None:
    stagnation = ScoreStagnation(tmp_path, 4)
    savestate = tmp_path / "Stage.state"
    savestate.write_bytes(b"state")

    assert stagnation.record("Stage", 10.0) == (0, False)
    for episode in range(1, 4):
        assert stagnation.record("Stage", 9.0) == (episode, False)
        assert savestate.exists()

    assert stagnation.record("Stage", 9.0) == (4, True)
    assert not savestate.exists()
