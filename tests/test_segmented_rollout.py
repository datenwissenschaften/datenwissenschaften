import numpy as np
import torch
from datenwissenschaften.segmented_rollout import SegmentedRecurrentRollouts, StateTransition, _advantages
from gymnasium import spaces
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


def test_recurrent_buffer_groups_interleaved_transitions_by_worker():
    zeros = torch.zeros((1, 2, 1))
    model = type(
        "Model",
        (),
        {
            "_last_lstm_states": RNNStates(pi=(zeros, zeros), vf=(zeros, zeros)),
            "observation_space": spaces.Dict(
                {"value": spaces.Box(low=-100.0, high=100.0, shape=(1,), dtype=np.float32)}
            ),
            "action_space": spaces.Discrete(2),
            "device": torch.device("cpu"),
            "gamma": 1.0,
            "gae_lambda": 1.0,
        },
    )()
    rollouts = SegmentedRecurrentRollouts({"State": model}, num_envs=2)
    interleaved = [
        transition(env=0, reward=1.0),
        transition(env=1, reward=10.0),
        transition(env=0, reward=2.0, segment_end=True),
        transition(env=1, reward=20.0, segment_end=True),
    ]
    for item in interleaved:
        item.observation = {"value": np.asarray([item.reward], dtype=np.float32)}
    rollouts.transitions["State"] = interleaved

    buffer = rollouts.build_buffer("State")

    assert buffer.observations["value"][:, 0, 0].tolist() == [1.0, 2.0, 10.0, 20.0]
    assert buffer.episode_starts[:, 0].tolist() == [1.0, 0.0, 1.0, 0.0]
    assert buffer.advantages[:, 0].tolist() == [3.0, 2.0, 30.0, 20.0]
