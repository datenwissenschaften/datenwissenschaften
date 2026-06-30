import neat
import numpy as np
from loguru import logger
from stable_baselines3.common.callbacks import BaseCallback

from datenwissenschaften import settings
from datenwissenschaften.neat.torch_network import TorchFeedForwardBatch
from datenwissenschaften.ui.control import model_reset_requested
from datenwissenschaften.ui.telemetry import publish_episode, publish_metadata


class NEATEvaluator:
    def __init__(
        self,
        env,
        *,
        training_state: str,
        controller_genomes: dict[str, object] | None = None,
        callback: BaseCallback | None = None,
    ):
        self.env = env
        self.training_state = training_state
        self.controller_genomes = dict(controller_genomes or {})
        self.callback = callback
        self.continue_training = True
        self.restart_requested = False
        self.generation_episodes_completed = 0
        self.generation_episodes_total = 0

    def evaluate_generation(self, genomes, config) -> None:
        genome_items = list(genomes)
        num_envs = self.env.num_envs
        self.generation_episodes_completed = 0
        self.generation_episodes_total = len(genome_items)
        self._publish_generation_progress()

        for _, genome in genome_items:
            genome.fitness = -10_000.0

        for start in range(0, len(genome_items), num_envs):
            batch = genome_items[start : start + num_envs]
            result = self.evaluate_batch(batch, config)

            if result is None:
                self.continue_training = False
                break

            for genome_id, genome in batch:
                genome.fitness = result[genome_id]

            highest_states = self.env.env_method("highest_progress_state")
            state_names = self.env.env_method("training_state_names")[0]
            training_index = state_names.index(self.training_state)

            if any(state_names.index(state_name) > training_index for state_name in highest_states):
                break

    def evaluate_batch(self, genomes, config) -> dict[int, float] | None:
        if model_reset_requested():
            self.restart_requested = True
            return None
        try:
            candidate_networks = [neat.nn.FeedForwardNetwork.create(genome, config) for _, genome in genomes]
            candidate_batch = TorchFeedForwardBatch.create(candidate_networks)
            controller_networks = {
                state_name: neat.nn.FeedForwardNetwork.create(genome, config)
                for state_name, genome in self.controller_genomes.items()
            }
        except KeyError as error:
            logger.warning(
                "Detected incompatible NEAT genome while creating networks "
                f"for state {self.training_state}: missing node {error}."
            )
            logger.warning("Deleting all model artifacts and stopping training.")
            settings.empty_all_paths()
            raise ValueError("Model input has changed. Model files were deleted. Please restart training.")

        restored = self.env.env_method("set_training_state", self.training_state)
        self.env.reset()

        if any(restored[: len(genomes)]):
            logger.debug(f"Restored automatic savestate for {self.training_state}")

        policy_inputs = self.env.env_method("policy_input")
        _, current_states = zip(*policy_inputs, strict=True)

        if candidate_batch is not None:
            logger.debug(f"Evaluating {len(genomes)} NEAT networks as a {candidate_batch.device.type} batch")

        active = [True] * len(genomes)

        training_steps = [0] * len(genomes)
        total_steps = [0] * len(genomes)

        episode_fitness = [0.0] * len(genomes)
        best_episode_fitness = [0.0] * len(genomes)

        while any(active):
            if model_reset_requested():
                self.restart_requested = True
                return None
            policy_inputs = self.env.env_method("policy_input")
            all_features, current_states = zip(*policy_inputs, strict=True)
            actions = []
            candidate_outputs = candidate_batch.activate(all_features[: len(genomes)]) if candidate_batch else None

            for i, is_active in enumerate(active):
                if not is_active:
                    actions.append(0)
                    continue

                state_name = current_states[i]

                if state_name == self.training_state:
                    if candidate_outputs is not None:
                        actions.append(int(np.argmax(candidate_outputs[i])))
                        continue
                    network = candidate_networks[i]
                else:
                    network = controller_networks.get(state_name)

                if network is None:
                    actions.append(0)
                else:
                    outputs = network.activate(all_features[i])
                    actions.append(int(np.argmax(outputs)))

            actions.extend([0] * (self.env.num_envs - len(actions)))

            _, rewards, dones, infos = self.env.step(actions)

            for i, is_active in enumerate(active):
                if not is_active:
                    continue

                state_before_step = current_states[i]
                total_steps[i] += 1

                if state_before_step == self.training_state:
                    reward = float(rewards[i])
                    episode_fitness[i] += reward
                    training_steps[i] += 1

                    if episode_fitness[i] > best_episode_fitness[i]:
                        best_episode_fitness[i] = episode_fitness[i]

                if bool(dones[i]):
                    active[i] = False

                    _, genome = genomes[i]
                    genome.fitness = best_episode_fitness[i]

                    self._report_episode(
                        env_index=i,
                        genome=genome,
                        training_steps=training_steps[i],
                        total_steps=total_steps[i],
                        info=infos[i],
                    )

            if self.callback is not None:
                self.callback.update_locals(
                    {
                        "rewards": rewards,
                        "dones": dones,
                        "infos": infos,
                    }
                )
                if not self.callback.on_step():
                    return None

        return {genome_id: best_episode_fitness[i] for i, (genome_id, _) in enumerate(genomes)}

    def _report_episode(
        self,
        *,
        env_index: int,
        genome,
        training_steps: int,
        total_steps: int,
        info: dict,
    ) -> None:
        episode = {
            "env": env_index,
            "training_state": self.training_state,
            "fitness": float(genome.fitness),
            "training_steps": training_steps,
            "total_steps": total_steps,
            "won": None if info.get("won") is None else bool(info.get("won")),
            "final_state": info.get("state"),
        }
        publish_episode(**episode)
        self.generation_episodes_completed += 1
        self._publish_generation_progress()
        logger.debug(
            "episode "
            f"env={env_index} "
            f"training_state={self.training_state} "
            f"fitness={genome.fitness:.2f} "
            f"training_steps={training_steps} "
            f"total_steps={total_steps} "
            f"won={info.get('won')} "
            f"final_state={info.get('state')}"
        )

    def _publish_generation_progress(self) -> None:
        publish_metadata(
            "neat",
            {
                "generation_episodes_completed": self.generation_episodes_completed,
                "generation_episodes_total": self.generation_episodes_total,
            },
        )
