from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from loguru import logger

from datenwissenschaften.accelerator import configure_accelerator


@dataclass(frozen=True)
class _Layer:
    target_indices: torch.Tensor
    weights: torch.Tensor
    biases: torch.Tensor
    responses: torch.Tensor
    mask: torch.Tensor


class TorchFeedForwardBatch:
    def __init__(self, networks, device: str) -> None:
        if not networks:
            raise ValueError("At least one network is required.")

        self.device = torch.device(device)
        # neat-python evaluates weights and activations as Python doubles.
        # Matching that precision avoids changing argmax tie-breaking for
        # saturated output nodes.
        self.dtype = torch.float64
        self.batch_size = len(networks)
        self.input_nodes = tuple(networks[0].input_nodes)
        self.output_nodes = tuple(networks[0].output_nodes)

        if any(tuple(network.input_nodes) != self.input_nodes for network in networks):
            raise ValueError("All networks must have the same input nodes.")
        if any(tuple(network.output_nodes) != self.output_nodes for network in networks):
            raise ValueError("All networks must have the same output nodes.")

        self._validate_operations(networks)
        node_keys = set(self.input_nodes) | set(self.output_nodes)
        for network in networks:
            for node, _, _, _, _, links in network.node_evals:
                node_keys.add(node)
                node_keys.update(source for source, _ in links)

        ordered_nodes = list(self.input_nodes) + sorted(node_keys.difference(self.input_nodes))
        node_indices = {node: index for index, node in enumerate(ordered_nodes)}
        self._node_count = len(ordered_nodes)
        self._input_indices = torch.tensor(
            [node_indices[node] for node in self.input_nodes], dtype=torch.long, device=self.device
        )
        self._output_indices = torch.tensor(
            [node_indices[node] for node in self.output_nodes], dtype=torch.long, device=self.device
        )
        self._layers = self._build_layers(networks, node_indices)

    @classmethod
    def create(cls, networks) -> TorchFeedForwardBatch | None:
        if not networks:
            return None
        device = configure_accelerator()
        if device != "cuda":
            return None
        try:
            return cls(networks, device)
        except (ValueError, torch.OutOfMemoryError) as error:
            logger.warning(f"Falling back to neat-python network evaluation: {error}")
            torch.cuda.empty_cache()
            return None

    def activate(self, inputs) -> np.ndarray:
        input_array = np.asarray(inputs, dtype=np.float64)
        expected_shape = (self.batch_size, len(self.input_nodes))
        if input_array.shape != expected_shape:
            raise RuntimeError(f"Expected input shape {expected_shape}, got {input_array.shape}.")

        with torch.inference_mode():
            values = torch.zeros(
                (self.batch_size, self._node_count),
                dtype=self.dtype,
                device=self.device,
            )
            values[:, self._input_indices] = torch.as_tensor(input_array, device=self.device)

            for layer in self._layers:
                sums = torch.bmm(layer.weights, values.unsqueeze(-1)).squeeze(-1)
                # neat-python's tanh activation scales by 2.5 and clamps the
                # input before applying tanh.
                activated = torch.tanh(torch.clamp(2.5 * (layer.biases + layer.responses * sums), -60.0, 60.0))
                current = values[:, layer.target_indices]
                values[:, layer.target_indices] = torch.where(layer.mask, activated, current)

            return values[:, self._output_indices].cpu().numpy()

    def _build_layers(self, networks, node_indices: dict[int, int]) -> tuple[_Layer, ...]:
        evaluations_by_depth = []
        maximum_depth = 0

        for network in networks:
            depths = {node: 0 for node in self.input_nodes}
            by_depth = {}
            for evaluation in network.node_evals:
                node, _, _, _, _, links = evaluation
                depth = max((depths[source] for source, _ in links), default=0) + 1
                depths[node] = depth
                by_depth.setdefault(depth, []).append(evaluation)
                maximum_depth = max(maximum_depth, depth)
            evaluations_by_depth.append(by_depth)

        layers = []
        for depth in range(1, maximum_depth + 1):
            targets = sorted(
                {evaluation[0] for by_depth in evaluations_by_depth for evaluation in by_depth.get(depth, [])}
            )
            if not targets:
                continue

            target_positions = {node: index for index, node in enumerate(targets)}
            shape = (self.batch_size, len(targets))
            weights = torch.zeros((*shape, self._node_count), dtype=self.dtype)
            biases = torch.zeros(shape, dtype=self.dtype)
            responses = torch.zeros(shape, dtype=self.dtype)
            mask = torch.zeros(shape, dtype=torch.bool)

            for network_index, by_depth in enumerate(evaluations_by_depth):
                for node, _, _, bias, response, links in by_depth.get(depth, []):
                    target = target_positions[node]
                    biases[network_index, target] = bias
                    responses[network_index, target] = response
                    mask[network_index, target] = True
                    for source, weight in links:
                        weights[network_index, target, node_indices[source]] = weight

            layers.append(
                _Layer(
                    target_indices=torch.tensor(
                        [node_indices[node] for node in targets], dtype=torch.long, device=self.device
                    ),
                    weights=weights.to(self.device),
                    biases=biases.to(self.device),
                    responses=responses.to(self.device),
                    mask=mask.to(self.device),
                )
            )

        return tuple(layers)

    @staticmethod
    def _validate_operations(networks) -> None:
        for network in networks:
            for _, activation, aggregation, _, _, _ in network.node_evals:
                if activation.__name__ != "tanh_activation" or aggregation.__name__ != "sum_aggregation":
                    raise ValueError("GPU NEAT evaluation supports only tanh activation and sum aggregation.")
