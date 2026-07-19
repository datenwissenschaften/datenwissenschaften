from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from gymnasium import spaces
from sb3_contrib.common.recurrent.buffers import RecurrentDictRolloutBuffer
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from stable_baselines3.common.utils import obs_as_tensor


@dataclass
class StateTransition:
    env_index: int
    observation: dict[str, np.ndarray]
    action: np.ndarray
    reward: float
    episode_start: bool
    segment_end: bool
    value: torch.Tensor
    log_prob: torch.Tensor
    next_value: float
    lstm_states: RNNStates


class SegmentedRecurrentRollouts:
    """Sparse recurrent rollouts separated by policy state and environment."""

    def __init__(self, models: dict[str, Any], num_envs: int) -> None:
        self.models = models
        self.num_envs = num_envs
        self.transitions: dict[str, list[StateTransition]] = {name: [] for name in models}
        self.lstm_states: dict[str, RNNStates] = {
            name: _clone_states(model._last_lstm_states) for name, model in models.items()
        }
        self.episode_starts: dict[str, np.ndarray] = {name: np.ones(num_envs, dtype=bool) for name in models}

    def actions(self, observations: dict[str, np.ndarray], state_names: list[str]) -> tuple[np.ndarray, dict[int, Any]]:
        action_space = next(iter(self.models.values())).action_space
        action_shape = (
            (self.num_envs, *action_space.shape) if isinstance(action_space, spaces.Box) else (self.num_envs,)
        )
        actions = np.zeros(action_shape, dtype=np.float32 if isinstance(action_space, spaces.Box) else np.int64)
        decisions: dict[int, Any] = {}

        for state_name in dict.fromkeys(state_names):
            model = self.models[state_name]
            model.policy.set_training_mode(False)
            indices = np.flatnonzero(np.asarray(state_names) == state_name)
            states = _select_states(self.lstm_states[state_name], indices)
            starts = self.episode_starts[state_name][indices]
            batch = {key: value[indices] for key, value in observations.items()}
            with torch.no_grad():
                policy_actions, values, log_probs, new_states = model.policy(
                    obs_as_tensor(batch, model.device),
                    states,
                    torch.as_tensor(starts, dtype=torch.float32, device=model.device),
                )
            policy_actions = policy_actions.cpu().numpy()
            if isinstance(action_space, spaces.Box):
                policy_actions = np.clip(policy_actions, action_space.low, action_space.high)
            else:
                policy_actions = policy_actions.reshape(-1)
            actions[indices] = policy_actions
            _replace_states(self.lstm_states[state_name], indices, new_states)
            for local_index, env_index in enumerate(indices):
                decisions[int(env_index)] = (
                    _single_states(states, local_index),
                    values[local_index : local_index + 1],
                    log_probs[local_index : local_index + 1],
                )
        return actions, decisions

    def append(
        self,
        observations: dict[str, np.ndarray],
        actions: np.ndarray,
        rewards: np.ndarray,
        new_observations: dict[str, np.ndarray],
        dones: np.ndarray,
        infos: list[dict[str, Any]],
        state_names: list[str],
        decisions: dict[int, Any],
        enabled_states: set[str] | None = None,
    ) -> set[str]:
        full: set[str] = set()
        for env_index, state_name in enumerate(state_names):
            if enabled_states is not None and state_name not in enabled_states:
                if dones[env_index]:
                    self.reset_environment(env_index)
                continue
            model = self.models[state_name]
            info = infos[env_index]
            segment_end = bool(info.get("state_segment_end", False) or dones[env_index])
            reward = float(rewards[env_index])
            if getattr(model, "rnd", None) is not None:
                reward_observation = {
                    key: np.expand_dims(value[env_index], axis=0) for key, value in new_observations.items()
                }
                intrinsic, coefficient = model.rnd.intrinsic_reward(
                    reward_observation,
                    np.asarray([segment_end], dtype=bool),
                )
                reward += float(coefficient * intrinsic[0])
                info["intrinsic_reward"] = float(intrinsic[0])
                info["intrinsic_reward_coefficient"] = float(coefficient)
                info["extrinsic_reward"] = float(rewards[env_index])

            old_states, value, log_prob = decisions[env_index]
            next_value = 0.0
            if not segment_end:
                next_batch = {key: np.expand_dims(item[env_index], axis=0) for key, item in new_observations.items()}
                next_states = _select_states(self.lstm_states[state_name], np.asarray([env_index]))
                with torch.no_grad():
                    predicted = model.policy.predict_values(
                        obs_as_tensor(next_batch, model.device),
                        next_states.vf,
                        torch.zeros(1, dtype=torch.float32, device=model.device),
                    )
                next_value = float(predicted.item())

            self.transitions[state_name].append(
                StateTransition(
                    env_index=env_index,
                    observation={key: np.array(item[env_index], copy=True) for key, item in observations.items()},
                    action=np.array(actions[env_index], copy=True),
                    reward=reward,
                    episode_start=bool(self.episode_starts[state_name][env_index]),
                    segment_end=segment_end,
                    value=value.detach(),
                    log_prob=log_prob.detach(),
                    next_value=next_value,
                    lstm_states=old_states,
                )
            )
            self.episode_starts[state_name][env_index] = segment_end
            if dones[env_index]:
                self.reset_environment(env_index)
            if len(self.transitions[state_name]) >= model.n_steps * model.n_envs:
                full.add(state_name)
        return full

    def build_buffer(self, state_name: str) -> RecurrentDictRolloutBuffer:
        model = self.models[state_name]
        transitions = self.transitions[state_name]
        count = len(transitions)
        if not count:
            raise ValueError(f"No transitions collected for state: {state_name}")
        first_hidden = transitions[0].lstm_states.pi[0]
        hidden_shape = (count, first_hidden.shape[0], 1, first_hidden.shape[-1])
        buffer = RecurrentDictRolloutBuffer(
            count,
            model.observation_space,
            model.action_space,
            hidden_shape,
            model.device,
            gamma=model.gamma,
            gae_lambda=model.gae_lambda,
            n_envs=1,
        )
        advantages = _advantages(transitions, gamma=model.gamma, gae_lambda=model.gae_lambda)
        # Transitions arrive interleaved across vector workers. A recurrent
        # buffer with n_envs=1 would otherwise interpret that interleaving as
        # one sequence, creating highly padded minibatches with as little as
        # one valid advantage. Group workers while retaining their chronology.
        order = sorted(range(count), key=lambda index: transitions[index].env_index)
        previous_env: int | None = None
        for index in order:
            transition = transitions[index]
            worker_boundary = transition.env_index != previous_env
            buffer.add(
                transition.observation,
                transition.action,
                np.asarray([transition.reward], dtype=np.float32),
                np.asarray([transition.episode_start or worker_boundary], dtype=np.float32),
                transition.value,
                transition.log_prob,
                lstm_states=transition.lstm_states,
            )
            previous_env = transition.env_index
        ordered_advantages = advantages[order]
        buffer.advantages[:, 0] = ordered_advantages
        buffer.returns[:, 0] = ordered_advantages + buffer.values[:, 0]
        self.transitions[state_name] = []
        return buffer

    def reset_environment(self, env_index: int) -> None:
        for state_name in self.models:
            self.episode_starts[state_name][env_index] = True
            _zero_environment_states(self.lstm_states[state_name], env_index)


def _advantages(transitions: list[StateTransition], *, gamma: float, gae_lambda: float) -> np.ndarray:
    advantages = np.zeros(len(transitions), dtype=np.float32)
    following: dict[int, tuple[float, bool]] = {}
    for index in range(len(transitions) - 1, -1, -1):
        transition = transitions[index]
        next_advantage, contiguous = following.get(transition.env_index, (0.0, False))
        non_terminal = 0.0 if transition.segment_end else 1.0
        delta = transition.reward + gamma * transition.next_value * non_terminal - float(transition.value.item())
        advantages[index] = delta + gamma * gae_lambda * non_terminal * next_advantage if contiguous else delta
        following[transition.env_index] = (float(advantages[index]), not transition.episode_start)
    return advantages


def _clone_states(states: RNNStates) -> RNNStates:
    return RNNStates(
        pi=tuple(item.detach().clone() for item in states.pi),
        vf=tuple(item.detach().clone() for item in states.vf),
    )


def _select_states(states: RNNStates, indices: np.ndarray) -> RNNStates:
    selected = indices.tolist()
    return RNNStates(
        pi=tuple(item[:, selected, :].contiguous() for item in states.pi),
        vf=tuple(item[:, selected, :].contiguous() for item in states.vf),
    )


def _single_states(states: RNNStates, index: int) -> RNNStates:
    return RNNStates(
        pi=tuple(item[:, index : index + 1, :].contiguous() for item in states.pi),
        vf=tuple(item[:, index : index + 1, :].contiguous() for item in states.vf),
    )


def _replace_states(target: RNNStates, indices: np.ndarray, source: RNNStates) -> None:
    selected = indices.tolist()
    for target_item, source_item in zip((*target.pi, *target.vf), (*source.pi, *source.vf), strict=True):
        target_item[:, selected, :] = source_item


def _zero_environment_states(states: RNNStates, env_index: int) -> None:
    for item in (*states.pi, *states.vf):
        item[:, env_index, :].zero_()
