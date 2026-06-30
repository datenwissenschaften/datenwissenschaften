from __future__ import annotations

from collections.abc import Callable
from typing import Any

import neat
import numpy as np

from datenwissenschaften.neat.torch_network import TorchFeedForwardBatch


class NEATPolicyRouter:
    def __init__(self, winners: dict[str, object]) -> None:
        self.winners = winners
        self.networks: dict[str, neat.nn.FeedForwardNetwork] = {}
        self.torch_batches: dict[tuple[str, ...], TorchFeedForwardBatch | None] = {}

    def predict(
        self,
        *,
        env: Any,
        config: Any,
        validate_input_size: Callable[[int], None],
    ) -> tuple[np.ndarray, None]:
        policy_inputs = env.env_method("policy_input")
        all_features, state_names = zip(*policy_inputs, strict=True)
        if all_features:
            validate_input_size(len(all_features[0]))

        actions = [0] * len(state_names)
        routed_networks = []
        for index, (state_name, features) in enumerate(zip(state_names, all_features, strict=True)):
            genome = self.winners.get(state_name)
            if genome is None:
                continue
            if state_name not in self.networks:
                self.networks[state_name] = neat.nn.FeedForwardNetwork.create(genome, config)
            routed_networks.append((index, state_name, features, self.networks[state_name]))

        batch_key = tuple(state_name for _, state_name, _, _ in routed_networks)
        if batch_key not in self.torch_batches:
            self.torch_batches[batch_key] = TorchFeedForwardBatch.create(
                [network for _, _, _, network in routed_networks]
            )

        torch_batch = self.torch_batches[batch_key]
        if torch_batch is not None:
            outputs = torch_batch.activate([features for _, _, features, _ in routed_networks])
            for (index, _, _, _), network_outputs in zip(routed_networks, outputs, strict=True):
                actions[index] = int(np.argmax(network_outputs))
        else:
            for index, _, features, network in routed_networks:
                actions[index] = int(np.argmax(network.activate(features)))
        return np.asarray(actions, dtype=np.int64), None
