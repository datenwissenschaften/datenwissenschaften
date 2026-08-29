from pathlib import Path

from datenwissenschaften.curriculum import ReverseCurriculum


def test_checkpoint_preserves_cumulative_score(tmp_path: Path) -> None:
    curriculum = ReverseCurriculum(tmp_path, ("First", "Second"))

    assert curriculum.save_checkpoint("Second", b"state", 10.0)
    assert curriculum.checkpoint("Second") == b"state"
    assert curriculum.checkpoint_score("Second") == 10.0


def test_checkpoint_survives_repeated_low_scores(tmp_path: Path) -> None:
    curriculum = ReverseCurriculum(tmp_path, ("First", "Second"))
    curriculum.save_checkpoint("Second", b"state", 10.0)
    curriculum.record_failure("Second", 100, 10.0)
    curriculum.record_failure("Second", 100, 9.0)

    for _ in range(curriculum.BAD_CHECKPOINT_EVIDENCE_TARGET - 2):
        curriculum.record_failure("Second", 100, 8.0)

    assert curriculum.has_checkpoint("Second")
    assert curriculum.checkpoint_score("Second") == 10.0
