from pathlib import Path

from datenwissenschaften.curriculum import ReverseCurriculum


def test_reverse_curriculum_starts_from_deepest_checkpoint_and_backtracks(tmp_path: Path):
    curriculum = ReverseCurriculum(tmp_path, ("Start", "Middle", "Finish"))
    curriculum.save_checkpoint("Middle", b"middle")
    curriculum.save_checkpoint("Finish", b"finish")

    assert curriculum.active_state() == "Finish"
    assert curriculum.record_success("Finish", 4) is False
    assert curriculum.record_success("Finish", 4) is True
    assert curriculum.active_state() == "Middle"
    assert curriculum.checkpoint("Middle") == b"middle"


def test_curriculum_requires_consecutive_successes(tmp_path: Path):
    curriculum = ReverseCurriculum(tmp_path, ("Start", "Finish"))

    curriculum.record_success("Finish", 16)
    curriculum.record_success("Finish", 16)
    curriculum.record_failure("Finish", 16)

    assert curriculum.successes("Finish") == 0
    assert curriculum.record_success("Finish", 16) is False


def test_initial_state_only_completes_after_success_threshold(tmp_path: Path):
    curriculum = ReverseCurriculum(tmp_path, ("Start", "Finish"))

    assert curriculum.record_success("Start", 1) is False
    assert curriculum.record_success("Start", 1) is True
    assert curriculum.progress()["Start"]["mastered"] is True


def test_bad_checkpoint_is_deleted_after_repeated_failures(tmp_path: Path):
    curriculum = ReverseCurriculum(tmp_path, ("Start", "Finish"))
    curriculum.save_checkpoint("Finish", b"unrecoverable")

    assert curriculum.record_failure("Finish", 100) is False
    assert curriculum.active_state() == "Finish"
    assert curriculum.record_failure("Finish", 1) is True
    assert curriculum.active_state() is None
    assert curriculum.failures("Finish") == 0


def test_bad_checkpoint_detector_never_deletes_initial_state(tmp_path: Path):
    curriculum = ReverseCurriculum(tmp_path, ("Start", "Finish"))

    assert curriculum.record_failure("Start", 1) is False
    assert curriculum.failures("Start") == 0


def test_dynamic_targets_scale_with_observed_episode_length(tmp_path: Path):
    short = ReverseCurriculum(tmp_path / "short", ("Start", "Finish"))
    long = ReverseCurriculum(tmp_path / "long", ("Start", "Finish"))

    short.record_success("Finish", 2)
    long.record_success("Finish", 256)

    assert short.success_threshold("Finish") < long.success_threshold("Finish")
    assert short.failure_threshold("Finish") < long.failure_threshold("Finish")


def test_success_target_is_relaxed_by_one_but_still_requires_repetition(tmp_path: Path):
    curriculum = ReverseCurriculum(tmp_path, ("Start", "Finish"))

    assert curriculum.success_threshold("Finish") == 2
    curriculum.record_success("Finish", 256)
    assert curriculum.success_threshold("Finish") == 8
