from pathlib import Path

import pytest

from datenwissenschaften.states.state import State
from datenwissenschaften.states.target import TargetState


def next_state(state: TargetState) -> type[TargetState] | None:
    return type(state)


def won(state: TargetState) -> bool:
    return True


def custom_reward(state: TargetState) -> float:
    return 1.0


def test_rejects_default_state(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="must inherit Explorer or TargetState"):
        State(tmp_path)


def test_rejects_custom_reward(tmp_path: Path) -> None:
    custom_state = type(
        "CustomReward",
        (TargetState,),
        {
            "template_file": "missing.png",
            "_next": next_state,
            "_reward": custom_reward,
        },
    )

    with pytest.raises(TypeError, match="cannot define custom rewards"):
        custom_state(tmp_path)


def test_target_requires_exactly_one_completion_outcome(tmp_path: Path) -> None:
    incomplete_state = type(
        "IncompleteTarget",
        (TargetState,),
        {"template_file": "missing.png"},
    )
    ambiguous_state = type(
        "AmbiguousTarget",
        (TargetState,),
        {
            "template_file": "missing.png",
            "_next": next_state,
            "_won": won,
        },
    )

    for state_type in (incomplete_state, ambiguous_state):
        with pytest.raises(TypeError, match="exactly one of next or won"):
            state_type(tmp_path)
