import neat
import numpy as np
from loguru import logger
from stable_baselines3.common.callbacks import BaseCallback

from datenwissenschaften import settings
from datenwissenschaften.neat.torch_network import TorchFeedForwardBatch
from datenwissenschaften.ui.telemetry import publish_episode


class NEATEvaluator:
    def __init__(
        self,
        env,
        *,
        training_state: str,
        controller_genomes: dict[str, object] | None = None,
        max_steps: int = 20_000,
        max_no_progress_steps: int = 600,
        episodes_per_genome: int = 3,
        callback: BaseCallback | None = None,
    ):
        self.env = env
        self.training_state = training_state
        self.controller_genomes = dict(controller_genomes or {})
        self.max_steps = max_steps
        self.max_no_progress_steps = max_no_progress_steps
        self.episodes_per_genome = episodes_per_genome
        self.callback = callback
        self.continue_training = True
        self.restart_requested = False

    def evaluate_generation(self, genomes, config) -> None:
        genome_items = list(genomes)
        num_envs = self.env.num_envs

        for _, genome in genome_items:
            genome.fitness = 0.0

        fitness_sums = {genome_id: 0.0 for genome_id, _ in genome_items}
        episode_counts = {genome_id: 0 for genome_id, _ in genome_items}

        for episode_index in range(self.episodes_per_genome):
            logger.debug(f"Evaluating NEAT episode pass {episode_index + 1}/{self.episodes_per_genome}")

            for start in range(0, len(genome_items), num_envs):
                batch = genome_items[start : start + num_envs]

                result = self.evaluate_batch(batch, config)

                if result is None:
                    self.continue_training = False
                    break

                for genome_id, episode_fitness in result.items():
                    fitness_sums[genome_id] += episode_fitness
                    episode_counts[genome_id] += 1

                highest_states = self.env.env_method("highest_progress_state")
                state_names = self.env.env_method("training_state_names")[0]
                training_index = state_names.index(self.training_state)

                if any(state_names.index(state_name) > training_index for state_name in highest_states):
                    break

            if not self.continue_training:
                break

        for genome_id, genome in genome_items:
            count = episode_counts[genome_id]
            if count == 0:
                genome.fitness = -10_000.0
            else:
                genome.fitness = fitness_sums[genome_id] / count

    def evaluate_batch(self, genomes, config) -> dict[int, float] | None:
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
        entered_training_state = [state_name == self.training_state for state_name in current_states[: len(genomes)]]

        training_steps = [0] * len(genomes)
        total_steps = [0] * len(genomes)

        episode_fitness = [0.0] * len(genomes)
        best_episode_fitness = [0.0] * len(genomes)
        steps_without_progress = [0] * len(genomes)

        while any(active):
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
                    entered_training_state[i] = True
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
                        steps_without_progress[i] = 0
                    else:
                        steps_without_progress[i] += 1

                state_after_step = infos[i].get("state")

                left_training_state = (
                    entered_training_state[i]
                    and state_before_step == self.training_state
                    and state_after_step != self.training_state
                )

                timed_out = total_steps[i] >= self.max_steps
                no_progress_timeout = steps_without_progress[i] >= self.max_no_progress_steps

                if bool(dones[i]) or left_training_state or timed_out or no_progress_timeout:
                    active[i] = False

                    _, genome = genomes[i]
                    genome.fitness = best_episode_fitness[i]

                    self._report_episode(
                        env_index=i,
                        genome=genome,
                        training_steps=training_steps[i],
                        total_steps=total_steps[i],
                        info=infos[i],
                        timed_out=timed_out,
                        no_progress_timeout=no_progress_timeout,
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
        timed_out: bool,
        no_progress_timeout: bool,
    ) -> None:
        episode = {
            "env": env_index,
            "training_state": self.training_state,
            "fitness": float(genome.fitness),
            "training_steps": training_steps,
            "total_steps": total_steps,
            "won": None if info.get("won") is None else bool(info.get("won")),
            "final_state": info.get("state"),
            "timed_out": timed_out,
            "no_progress_timeout": no_progress_timeout,
        }
        publish_episode(**episode)
        logger.debug(
            "episode "
            f"env={env_index} "
            f"training_state={self.training_state} "
            f"fitness={genome.fitness:.2f} "
            f"training_steps={training_steps} "
            f"total_steps={total_steps} "
            f"won={info.get('won')} "
            f"final_state={info.get('state')} "
            f"timed_out={timed_out} "
            f"no_progress_timeout={no_progress_timeout}"
        )
