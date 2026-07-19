from pathlib import Path

from datenwissenschaften.curriculum import ReverseCurriculum


def test_reverse_curriculum_starts_from_deepest_checkpoint_and_backtracks(tmp_path: Path):
    curriculum = ReverseCurriculum(tmp_path, ("Start", "Middle", "Finish"))
    curriculum.save_checkpoint("Middle", b"middle")
    curriculum.save_checkpoint("Finish", b"finish")

    assert curriculum.active_state() == "Finish"
    for _ in range(7):
        assert curriculum.record_success("Finish", 4) is False
    assert curriculum.record_success("Finish", 4) is True
    assert curriculum.active_state() == "Middle"
    assert curriculum.checkpoint("Middle") == b"middle"


def test_curriculum_counts_wins_across_failures(tmp_path: Path):
    curriculum = ReverseCurriculum(tmp_path, ("Start", "Finish"))

    curriculum.record_success("Finish", 16)
    curriculum.record_success("Finish", 16)
    curriculum.record_failure("Finish", 16, 10.0)

    assert curriculum.wins("Finish") == 2
    assert curriculum.record_success("Finish", 16) is False
    assert curriculum.wins("Finish") == 3


def test_initial_state_only_completes_after_success_threshold(tmp_path: Path):
    curriculum = ReverseCurriculum(tmp_path, ("Start", "Finish"))

    for _ in range(7):
        assert curriculum.record_success("Start", 1) is False
    assert curriculum.record_success("Start", 1) is True
    assert curriculum.progress()["Start"]["mastered"] is True


def test_bad_checkpoint_is_deleted_after_persistent_score_stagnation(tmp_path: Path):
    curriculum = ReverseCurriculum(tmp_path, ("Start", "Finish"))
    curriculum.save_checkpoint("Finish", b"unrecoverable")

    assert curriculum.record_failure("Finish", 100, 10.0) is False
    for _ in range(31):
        assert curriculum.record_failure("Finish", 100, 10.0) is False
    assert curriculum.record_failure("Finish", 100, 10.0) is True
    assert curriculum.active_state() is None
    assert curriculum.stagnation_evidence("Finish") == 0


def test_score_improvement_resets_stagnation_evidence(tmp_path: Path):
    curriculum = ReverseCurriculum(tmp_path, ("Start", "Finish"))
    curriculum.save_checkpoint("Finish", b"recoverable")

    curriculum.record_failure("Finish", 100, 10.0)
    curriculum.record_failure("Finish", 100, 10.0)
    assert curriculum.stagnation_evidence("Finish") == 1

    curriculum.record_failure("Finish", 100, 11.0)
    assert curriculum.stagnation_evidence("Finish") == 0
    assert curriculum.best_score("Finish") == 11.0


def test_bad_checkpoint_evidence_target_allows_extended_stagnation(tmp_path: Path):
    curriculum = ReverseCurriculum(tmp_path, ("Start", "Finish"))

    assert curriculum.bad_checkpoint_evidence_target("Finish") == 32
    assert curriculum.progress()["Finish"]["bad_checkpoint_evidence_target"] == 32


def test_declining_scores_accumulate_evidence_twice_as_fast(tmp_path: Path):
    curriculum = ReverseCurriculum(tmp_path, ("Start", "Finish"))
    curriculum.save_checkpoint("Finish", b"declining")

    curriculum.record_failure("Finish", 100, 10.0)
    curriculum.record_failure("Finish", 100, 9.0)
    assert curriculum.stagnation_evidence("Finish") == 2
    curriculum.record_failure("Finish", 100, 8.0)
    assert curriculum.stagnation_evidence("Finish") == 4


def test_bad_checkpoint_detector_never_deletes_initial_state(tmp_path: Path):
    curriculum = ReverseCurriculum(tmp_path, ("Start", "Finish"))

    assert curriculum.record_failure("Start", 1, 0.0) is False
    assert curriculum.stagnation_evidence("Start") == 0


def test_success_target_is_always_eight(tmp_path: Path):
    curriculum = ReverseCurriculum(tmp_path, ("Start", "Finish"))

    assert curriculum.win_target("Finish") == 8
    curriculum.record_success("Finish", 256)
    assert curriculum.win_target("Finish") == 8
    assert curriculum.progress()["Finish"]["wins"] == 1
    assert curriculum.progress()["Finish"]["win_target"] == 8
