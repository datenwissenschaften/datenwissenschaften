from pathlib import Path

from datenwissenschaften.curriculum import ReverseCurriculum


def test_reverse_curriculum_starts_from_deepest_checkpoint_and_backtracks(tmp_path: Path):
    curriculum = ReverseCurriculum(tmp_path, ("Start", "Middle", "Finish"), success_threshold=3)
    curriculum.save_checkpoint("Middle", b"middle")
    curriculum.save_checkpoint("Finish", b"finish")

    assert curriculum.active_state() == "Finish"
    assert curriculum.record_success("Finish") is False
    assert curriculum.record_success("Finish") is False
    assert curriculum.record_success("Finish") is True
    assert curriculum.active_state() == "Middle"
    assert curriculum.checkpoint("Middle") == b"middle"


def test_curriculum_requires_consecutive_successes(tmp_path: Path):
    curriculum = ReverseCurriculum(tmp_path, ("Start", "Finish"), success_threshold=3)

    curriculum.record_success("Finish")
    curriculum.record_success("Finish")
    curriculum.record_failure("Finish")

    assert curriculum.successes("Finish") == 0
    assert curriculum.record_success("Finish") is False


def test_initial_state_only_completes_after_success_threshold(tmp_path: Path):
    curriculum = ReverseCurriculum(tmp_path, ("Start", "Finish"), success_threshold=2)

    assert curriculum.record_success("Start") is False
    assert curriculum.record_success("Start") is True
    assert curriculum.progress()["Start"]["mastered"] is True
