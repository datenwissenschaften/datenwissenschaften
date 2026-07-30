from dataclasses import make_dataclass
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from datenwissenschaften.gym.wrapper import StateMachineGymWrapper, _observation_space, _shape_reward
from datenwissenschaften.ram.model import REQUIRED_DQN_RAM_FIELDS, RamInfo, ram
from datenwissenschaften.states.state import State


def ram_info(fields: tuple[str, ...]) -> type[RamInfo]:
    definitions = [(name, int, ram(address)) for address, name in enumerate(fields)]
    return make_dataclass("TestRam", definitions, bases=(RamInfo,), slots=True)


def test_accepts_all_required_dqn_ram_fields() -> None:
    space = _observation_space(
        ram_info(REQUIRED_DQN_RAM_FIELDS),
        (State,),
    )

    assert space.shape == (10,)


def test_rejects_missing_dqn_ram_fields() -> None:
    with pytest.raises(ValueError, match=r"ram\.player_y"):
        _observation_space(
            ram_info(REQUIRED_DQN_RAM_FIELDS[:-1]),
            (State,),
        )


def test_rejects_non_finite_state_reward() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        _shape_reward(float("nan"))


def test_failure_adds_penalty(monkeypatch: pytest.MonkeyPatch) -> None:
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
    wrapper.episode_number = 1
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
        lambda ram, states, current, velocity: np.zeros(1, dtype=np.float32),
    )

    _, reward, _, truncated, info = wrapper.step(0)

    assert truncated
    assert reward == -5.01
    assert info["action_repeat"] == 1


def test_transition_continues_episode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper = object.__new__(StateMachineGymWrapper)
    wrapper.action_repeat = 4
    wrapper.action_table = np.asarray([[0]], dtype=np.int8)
    wrapper.action_space = Mock()
    wrapper.action_space.contains.return_value = True
    wrapper.ram_info_cls = Mock()
    wrapper.env = Mock()
    wrapper.env.step.return_value = np.zeros((1, 1, 3)), 0.0, False, False, {}
    wrapper.env.unwrapped = SimpleNamespace(
        movie_path=".",
        gamename="Game",
        statename="State",
        movie_id=1,
    )
    wrapper.machine = Mock()
    wrapper.machine.name = "First"

    def transition(ram: object, frame: np.ndarray) -> tuple[float, bool, bool]:
        wrapper.machine.name = "Second"
        return 2.0, False, False

    wrapper.machine.step.side_effect = transition
    wrapper.machine.current._won.return_value = False
    wrapper.episode_number = 1
    wrapper.transition_reward = 5.0
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
        lambda ram, states, current, velocity: np.zeros(1, dtype=np.float32),
    )

    _, reward, terminated, truncated, _ = wrapper.step(0)

    assert reward == 5.99
    assert not terminated
    assert not truncated
    wrapper.env.step.assert_called_once()


def test_bounds_custom_reward_before_adding_transition_reward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper = object.__new__(StateMachineGymWrapper)
    wrapper.action_repeat = 1
    wrapper.action_table = np.asarray([[0]], dtype=np.int8)
    wrapper.action_space = Mock()
    wrapper.action_space.contains.return_value = True
    wrapper.ram_info_cls = Mock()
    wrapper.env = Mock()
    wrapper.env.step.return_value = np.zeros((1, 1, 3)), 0.0, False, False, {}
    wrapper.env.unwrapped = SimpleNamespace()
    wrapper.machine = Mock()
    wrapper.machine.name = "First"

    def transition(ram: object, frame: np.ndarray) -> tuple[float, bool, bool]:
        wrapper.machine.name = "Second"
        return 10_000.0, False, False

    wrapper.machine.step.side_effect = transition
    wrapper.machine.current._won.return_value = False
    wrapper.episode_number = 1
    wrapper.transition_reward = 5.0
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
        lambda ram, states, current, velocity: np.zeros(1, dtype=np.float32),
    )

    _, reward, _, _, _ = wrapper.step(0)

    assert reward == 5.99


def test_does_not_reward_a_transition_on_a_failed_frame(
    monkeypatch: pytest.MonkeyPatch,
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
        movie_path=".",
        gamename="Game",
        statename="State",
        movie_id=1,
    )
    wrapper.machine = Mock()
    wrapper.machine.name = "First"

    def failed_transition(ram: object, frame: np.ndarray) -> tuple[float, bool, bool]:
        wrapper.machine.name = "Second"
        return 0.0, False, True

    wrapper.machine.step.side_effect = failed_transition
    wrapper.machine.current._won.return_value = False
    wrapper.episode_number = 1
    wrapper.transition_reward = 5.0
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
        lambda ram, states, current, velocity: np.zeros(1, dtype=np.float32),
    )

    _, reward, _, truncated, _ = wrapper.step(0)

    assert truncated
    assert reward == -5.01
