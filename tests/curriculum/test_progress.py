from pathlib import Path

from datenwissenschaften.curriculum.progress import SavestateCurriculum


def curriculum(root: Path, full_run_probability: float) -> SavestateCurriculum:
    return SavestateCurriculum(root, ("Start", "Finish"), 3, 256, full_run_probability)


def test_requires_repeated_success_before_advancing(tmp_path: Path) -> None:
    progress = curriculum(tmp_path, 1.0)
    for success in range(1, 4):
        assert progress.start() == ("Start", None)
        assert progress.transition("Start", "Finish", b"finish") == (success, success == 3)
    assert progress.start() == ("Finish", b"finish")


def test_completed_curriculum_starts_full_runs(tmp_path: Path) -> None:
    progress = curriculum(tmp_path, 1.0)
    progress.storage.complete("Start")
    progress.storage.complete("Finish")
    progress.storage.save("Finish", b"finish")
    assert progress.start() == ("Start", None)
    assert progress.full_run
    assert progress.transition("Start", "Finish", b"new") == (0, False)


def test_completed_curriculum_keeps_practising_intermediate_states(tmp_path: Path) -> None:
    progress = curriculum(tmp_path, 0.0)
    progress.storage.complete("Start")
    progress.storage.complete("Finish")
    progress.storage.save("Finish", b"finish")
    assert progress.start() == ("Finish", b"finish")
    assert not progress.full_run
    assert not progress.training


def test_success_resets_stagnation_for_current_state(tmp_path: Path) -> None:
    progress = curriculum(tmp_path, 1.0)
    progress.storage.complete("Start")
    progress.storage.save("Finish", b"finish")

    assert progress.start() == ("Finish", b"finish")
    assert progress.record_attempt(10.0, False).attempts == 1
    assert progress.start() == ("Finish", b"finish")
    assert progress.record_attempt(9.0, False).attempts == 2
    assert progress.start() == ("Finish", b"finish")
    assert progress.victory("Finish") == (1, False)
    assert not (tmp_path / "stagnation" / "Finish.json").exists()
