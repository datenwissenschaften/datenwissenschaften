import neat
import numpy as np
from loguru import logger
from stable_baselines3.common.callbacks import BaseCallback

from datenwissenschaften import settings


class NEATEvaluator:
    def __init__(
        self,
        env,
        *,
        training_state: str,
        controller_genomes: dict[str, object] | None = None,
        max_steps: int = 20_000,
        callback: BaseCallback | None = None,
    ):
        self.env = env
        self.training_state = training_state
        self.controller_genomes = dict(controller_genomes or {})
        self.max_steps = max_steps
        self.callback = callback
        self.continue_training = True
        self.restart_requested = False

    def evaluate_generation(self, genomes, config) -> None:
        genome_items = list(genomes)
        num_envs = self.env.num_envs
        for _, genome in genome_items:
            genome.fitness = 0.0

        for start in range(0, len(genome_items), num_envs):
            batch = genome_items[start : start + num_envs]
            if not self.evaluate_batch(batch, config):
                self.continue_training = False
                break
            highest_states = self.env.env_method("highest_progress_state")
            state_names = self.env.env_method("training_state_names")[0]
            training_index = state_names.index(self.training_state)
            if any(state_names.index(state_name) > training_index for state_name in highest_states):
                break

    def evaluate_batch(self, genomes, config) -> bool:
        try:
            candidate_networks = [neat.nn.FeedForwardNetwork.create(genome, config) for _, genome in genomes]
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
        current_states = self.env.env_method("state_name")
        active = [True] * len(genomes)
        entered_training_state = [state_name == self.training_state for state_name in current_states[: len(genomes)]]
        training_steps = [0] * len(genomes)
        total_steps = [0] * len(genomes)

        while any(active):
            all_features = self.env.env_method("features")
            current_states = self.env.env_method("state_name")
            actions = []

            for i, is_active in enumerate(active):
                if not is_active:
                    actions.append(0)
                    continue

                state_name = current_states[i]
                if state_name == self.training_state:
                    entered_training_state[i] = True
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

                _, genome = genomes[i]
                state_before_step = current_states[i]
                total_steps[i] += 1

                if state_before_step == self.training_state:
                    genome.fitness += float(rewards[i])
                    training_steps[i] += 1

                state_after_step = infos[i].get("state")
                left_training_state = (
                    entered_training_state[i]
                    and state_before_step == self.training_state
                    and state_after_step != self.training_state
                )
                timed_out = total_steps[i] >= self.max_steps

                if bool(dones[i]) or left_training_state or timed_out:
                    active[i] = False
                    self._report_episode(
                        env_index=i,
                        genome=genome,
                        training_steps=training_steps[i],
                        total_steps=total_steps[i],
                        info=infos[i],
                        timed_out=timed_out,
                    )

            if self.callback is not None:
                self.callback.update_locals({"rewards": rewards, "dones": dones, "infos": infos})
                if not self.callback.on_step():
                    return False

        return True

    def _report_episode(
        self,
        *,
        env_index: int,
        genome,
        training_steps: int,
        total_steps: int,
        info: dict,
        timed_out: bool,
    ) -> None:
        logger.info(
            "episode "
            f"env={env_index} "
            f"training_state={self.training_state} "
            f"fitness={genome.fitness:.2f} "
            f"training_steps={training_steps} "
            f"total_steps={total_steps} "
            f"score={info.get('score')} "
            f"won={info.get('won')} "
            f"final_state={info.get('state')} "
            f"timed_out={timed_out}"
        )
