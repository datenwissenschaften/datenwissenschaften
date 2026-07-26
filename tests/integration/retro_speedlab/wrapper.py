from pathlib import Path

import gymnasium as gym
import numpy as np
from datenwissenschaften.gym.wrapper import StateMachineGymWrapper

from tests.integration.retro_speedlab.done import Done
from tests.integration.retro_speedlab.explore import ExploreTarget
from tests.integration.retro_speedlab.ram import AirstrikerRam
from tests.integration.retro_speedlab.target import ApproachTarget

GENESIS_BUTTONS = 12
FIRE, UP, DOWN, LEFT, RIGHT = 0, 4, 5, 6, 7
ACTIONS = (
    (),
    (FIRE,),
    (UP, FIRE),
    (DOWN, FIRE),
    (LEFT, FIRE),
    (RIGHT, FIRE),
    (UP, LEFT, FIRE),
    (UP, RIGHT, FIRE),
    (DOWN, LEFT, FIRE),
    (DOWN, RIGHT, FIRE),
)


def action_table() -> np.ndarray:
    table = np.zeros((len(ACTIONS), GENESIS_BUTTONS), dtype=np.int8)
    for index, buttons in enumerate(ACTIONS):
        table[index, buttons] = 1
    return table


class AirstrikerWrapper(StateMachineGymWrapper[AirstrikerRam]):
    ram_info_cls = AirstrikerRam
    start_state_cls = ExploreTarget
    training_state_classes = (ExploreTarget, ApproachTarget, Done)
    action_repeat = 1
    grayscale = False

    def __init__(self, environment: gym.Env, *, model_dir: Path) -> None:
        super().__init__(environment, obs_size=(96, 96), action_table=action_table(), model_dir=model_dir)
