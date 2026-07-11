import torch
from datenwissenschaften.segmented_rollout import StateTransition, _advantages
from sb3_contrib.common.recurrent.type_aliases import RNNStates


def transition(
    *,
    env: int,
    reward: float,
    value: float = 0.0,
    next_value: float = 0.0,
    episode_start: bool = False,
    segment_end: bool = False,
) -> StateTransition:
    zeros = torch.zeros((1, 1, 1))
    states = RNNStates(pi=(zeros, zeros), vf=(zeros, zeros))
    return StateTransition(
        env_index=env,
        observation={},
        action=torch.zeros(1).numpy(),
        reward=reward,
        episode_start=episode_start,
        segment_end=segment_end,
        value=torch.tensor([value]),
        log_prob=torch.zeros(1),
        next_value=next_value,
        lstm_states=states,
    )


def test_advantages_are_independent_between_vector_environments():
    transitions = [
        transition(env=0, reward=1.0, episode_start=True),
        transition(env=1, reward=10.0, episode_start=True),
        transition(env=0, reward=2.0, segment_end=True),
        transition(env=1, reward=20.0, segment_end=True),
    ]

    result = _advantages(transitions, gamma=1.0, gae_lambda=1.0)

    assert result.tolist() == [3.0, 30.0, 2.0, 20.0]


def test_advantages_do_not_cross_state_segment_boundaries():
    transitions = [
        transition(env=0, reward=1.0, episode_start=True, segment_end=True),
        transition(env=0, reward=5.0, episode_start=True, segment_end=True),
    ]

    result = _advantages(transitions, gamma=1.0, gae_lambda=1.0)

    assert result.tolist() == [1.0, 5.0]
