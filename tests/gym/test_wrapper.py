from dataclasses import make_dataclass
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from datenwissenschaften.curriculum.progress import SavestateDiagnostics
from datenwissenschaften.gym.wrapper import StateMachineGymWrapper, _observation_space
from datenwissenschaften.ram.model import REQUIRED_DQN_RAM_FIELDS, RamInfo, ram


def ram_info(fields: tuple[str, ...]) -> type[RamInfo]:
    definitions = [(name, int, ram(address)) for address, name in enumerate(fields)]
    return make_dataclass("TestRam", definitions, bases=(RamInfo,), slots=True)


def test_accepts_all_required_dqn_ram_fields() -> None:
    space = _observation_space(ram_info(REQUIRED_DQN_RAM_FIELDS))

    assert space.shape == (9,)


def test_rejects_missing_dqn_ram_fields() -> None:
    with pytest.raises(ValueError, match=r"ram\.player_y"):
        _observation_space(ram_info(REQUIRED_DQN_RAM_FIELDS[:-1]))


def test_records_failed_curriculum_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    wrapper = object.__new__(StateMachineGymWrapper)
    wrapper.action_repeat = 1
    wrapper.action_table = np.asarray([[0]], dtype=np.int8)
    wrapper.action_space = Mock()
    wrapper.action_space.contains.return_value = True
    wrapper.ram_info_cls = Mock()
    wrapper.env = Mock()
    wrapper.env.step.return_value = np.zeros((1, 1, 3)), 0.0, False, True, {}
    wrapper.env.unwrapped = SimpleNamespace(
        movie_path=".",
        gamename="Game",
        statename="State",
        movie_id=1,
    )
    wrapper.machine = Mock()
    wrapper.machine.name = "Stage"
    wrapper.machine.step.return_value = 0.0, False, False
    wrapper.machine.current._won.return_value = False
    wrapper.curriculum = Mock()
    wrapper.curriculum.recorded = False
    wrapper.curriculum.episode_state = "Stage"
    wrapper.curriculum.full_run = False
    wrapper.curriculum.record_attempt.return_value = SavestateDiagnostics(
        "Stage",
        1,
        0.0,
        0.0,
        0.0,
        0.0,
        0,
        False,
    )
    wrapper.initial_episode_state = "Stage"
    wrapper.episode_number = 1
    wrapper.episode_score = 0.0
    wrapper.failure_penalty = -5.0
    wrapper.player_motion = Mock()
    wrapper.player_motion.measure.return_value = np.zeros(2, dtype=np.float32)
    wrapper.state_types = ()
    monkeypatch.setattr(
        "datenwissenschaften.gym.wrapper._ram",
        lambda model, emulator: SimpleNamespace(to_dict=lambda: {}),
    )
    monkeypatch.setattr(
        "datenwissenschaften.gym.wrapper._observation",
        lambda ram, current, velocity: np.zeros(1, dtype=np.float32),
    )

    _, _, _, truncated, info = wrapper.step(0)

    assert truncated
    assert info["action_repeat"] == 1
    wrapper.curriculum.record_attempt.assert_called_once_with(-5.0, False)


@pytest.mark.parametrize(
    ("full_run", "expected_terminated"),
    ((False, True), (True, False)),
)
def test_transition_ends_only_curriculum_episode(
    monkeypatch: pytest.MonkeyPatch,
    full_run: bool,
    expected_terminated: bool,
) -> None:
    wrapper = object.__new__(StateMachineGymWrapper)
    wrapper.action_repeat = 1
    wrapper.action_table = np.asarray([[0]], dtype=np.int8)
    wrapper.action_space = Mock()
    wrapper.action_space.contains.return_value = True
    wrapper.ram_info_cls = Mock()
    wrapper.env = Mock()
    wrapper.env.step.return_value = np.zeros((1, 1, 3)), 0.0, False, False, {}
    wrapper.env.unwrapped = SimpleNamespace(
        em=Mock(),
        movie_path=".",
        gamename="Game",
        statename="State",
        movie_id=1,
    )
    wrapper.env.unwrapped.em.get_state.return_value = b"state"
    wrapper.machine = Mock()
    wrapper.machine.name = "First"

    def transition(ram: object, frame: np.ndarray) -> tuple[float, bool, bool]:
        wrapper.machine.name = "Second"
        return 2.0, False, False

    wrapper.machine.step.side_effect = transition
    wrapper.machine.current._won.return_value = False
    wrapper.curriculum = Mock()
    wrapper.curriculum.recorded = True
    wrapper.curriculum.episode_state = "First"
    wrapper.curriculum.full_run = full_run
    wrapper.curriculum.transition.return_value = 0, False
    wrapper.initial_episode_state = "First"
    wrapper.episode_number = 1
    wrapper.episode_score = 0.0
    wrapper.transition_reward = 5.0
    wrapper.failure_penalty = -5.0
    wrapper.player_motion = Mock()
    wrapper.player_motion.measure.return_value = np.zeros(2, dtype=np.float32)
    monkeypatch.setattr(
        "datenwissenschaften.gym.wrapper._ram",
        lambda model, emulator: SimpleNamespace(to_dict=lambda: {}),
    )
    monkeypatch.setattr(
        "datenwissenschaften.gym.wrapper._observation",
        lambda ram, current, velocity: np.zeros(1, dtype=np.float32),
    )

    _, reward, terminated, truncated, _ = wrapper.step(0)

    assert reward == 7.0
    assert terminated is expected_terminated
    assert not truncated
