import numpy as np
import pytest

from datenwissenschaften.states.ram_scorer import RamScorerState


class MockRam:
    def __init__(self):
        self.screen_x = 0
        self.screen_y = 0
        self.player_x = 0
        self.player_y = 0
        self.score = 0


class ConcreteRamScorer(RamScorerState):
    def _scored_value(self) -> float:
        return float(self.ram.score)

    def _won(self) -> bool:
        return True


@pytest.fixture
def ram_scorer(tmp_path):
    state = ConcreteRamScorer.__new__(ConcreteRamScorer)
    state.model_dir = tmp_path
    state.frame = np.zeros((100, 100, 3), dtype=np.uint8)
    state.ram = MockRam()
    state._on_reset()
    return state


def test_ram_scorer_rewards_increase(ram_scorer):
    ram_scorer.ram.score = 10
    reward, _, _, _ = ram_scorer.step(ram_scorer.ram, ram_scorer.frame)
    assert reward == 10.0


def test_ram_scorer_punishes_decrease(ram_scorer):
    ram_scorer.ram.score = 10
    ram_scorer.step(ram_scorer.ram, ram_scorer.frame)

    ram_scorer.ram.score = 5
    reward, _, _, _ = ram_scorer.step(ram_scorer.ram, ram_scorer.frame)
    assert reward == -5.0


def test_ram_scorer_zero_reward_when_unchanged(ram_scorer):
    ram_scorer.ram.score = 10
    ram_scorer.step(ram_scorer.ram, ram_scorer.frame)

    reward, _, _, _ = ram_scorer.step(ram_scorer.ram, ram_scorer.frame)
    assert reward == 0.0
