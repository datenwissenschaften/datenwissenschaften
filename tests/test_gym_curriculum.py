from pathlib import Path
from types import SimpleNamespace

import numpy as np
from datenwissenschaften.gym import StateMachineGymWrapper


class StateReceiver:
    def __init__(self):
        self.states = []

    def set_state(self, state):
        self.states.append(state)


def test_automatic_savestate_is_embedded_in_active_movie():
    emulator_state = b"automatic checkpoint"
    em = StateReceiver()
    movie = StateReceiver()
    data_calls = []
    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    emulator = SimpleNamespace(
        em=em,
        movie=movie,
        data=SimpleNamespace(
            reset=lambda: data_calls.append("reset"),
            update_ram=lambda: data_calls.append("update_ram"),
        ),
        get_screen=lambda *, apply_rotation: frame,
    )
    wrapper = SimpleNamespace(env=SimpleNamespace(unwrapped=emulator))

    restored_frame = StateMachineGymWrapper._restore_automatic_savestate(wrapper, emulator_state)

    assert em.states == [emulator_state]
    assert movie.states == [emulator_state]
    assert data_calls == ["reset", "update_ram"]
    assert restored_frame is frame


def test_automatic_savestate_restore_allows_disabled_movie_recording():
    emulator_state = b"automatic checkpoint"
    em = StateReceiver()
    emulator = SimpleNamespace(
        em=em,
        movie=None,
        data=SimpleNamespace(reset=lambda: None, update_ram=lambda: None),
        get_screen=lambda *, apply_rotation: np.zeros((1, 1, 3), dtype=np.uint8),
    )
    wrapper = SimpleNamespace(env=SimpleNamespace(unwrapped=emulator))

    StateMachineGymWrapper._restore_automatic_savestate(wrapper, emulator_state)

    assert em.states == [emulator_state]


def test_active_movie_path_uses_stable_retro_movie_id(tmp_path: Path):
    emulator = SimpleNamespace(
        movie_path=str(tmp_path),
        movie_id=12,
        gamename="Game",
        statename="nested/Level1.state",
    )
    wrapper = SimpleNamespace(env=SimpleNamespace(unwrapped=emulator))

    path = StateMachineGymWrapper._active_movie_path(wrapper)

    assert path == str(tmp_path / "Game-Level1-000011.bk2")


def test_movie_directory_is_recreated_after_artifact_cleanup(tmp_path: Path):
    movie_path = tmp_path / "recordings" / "Game" / "Level1" / "2"
    emulator = SimpleNamespace(movie_path=str(movie_path))
    wrapper = SimpleNamespace(env=SimpleNamespace(unwrapped=emulator))

    StateMachineGymWrapper._ensure_movie_directory(wrapper)

    assert movie_path.is_dir()


def test_movie_directory_repair_allows_disabled_recording(tmp_path: Path):
    emulator = SimpleNamespace(movie_path=None)
    wrapper = SimpleNamespace(env=SimpleNamespace(unwrapped=emulator))

    StateMachineGymWrapper._ensure_movie_directory(wrapper)

    assert list(tmp_path.iterdir()) == []
