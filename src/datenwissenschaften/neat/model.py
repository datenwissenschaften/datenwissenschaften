import hashlib
import inspect
import math
import pickle
from copy import copy, deepcopy
from itertools import count
from pathlib import Path

import neat
import numpy as np
from loguru import logger
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

from datenwissenschaften import settings
from datenwissenschaften.logger import setup_logging
from datenwissenschaften.neat.checkpointer import AtomicCheckpointer
from datenwissenschaften.neat.config import write_neat_config
from datenwissenschaften.neat.evaluator import NEATEvaluator
from datenwissenschaften.neat.reporter import AdaptiveConfigReporter, LoguruReporter, WinnerReporter
from datenwissenschaften.neat.torch_network import TorchFeedForwardBatch
from datenwissenschaften.runtime import get_runtime
from datenwissenschaften.settings import DEFAULT_CONFIG_PATH, load_config
from datenwissenschaften.vision.encoder import FixedVisualEncoder


class NEATModel:
    def __init__(
        self,
        env,
        generations_completed: dict[str, int] | None = None,
        winners: dict[str, object] | None = None,
        winner=None,
        input_signature: dict[str, int] | None = None,
        settings_path: str | Path = DEFAULT_CONFIG_PATH,
    ):
        settings = load_config(settings_path)
        setup_logging(settings.log_level)
        self.env = env
        game = settings.training.game
        self.output_dir = settings.paths.models_dir / game / "datenwissenschaften"
        self.config_path = self.output_dir / "config.ini"
        self.population_size = settings.training.population_size
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.generations_completed = dict(generations_completed or {})
        self.winners = dict(winners or {})
        self.winner = winner
        self.input_signature = dict(input_signature or {})
        self.populations: dict[str, neat.Population] = {}
        self.population = None
        self.statistics: dict[str, neat.StatisticsReporter] = {}
        self._networks: dict[str, neat.nn.FeedForwardNetwork] = {}
        self._torch_network_batches: dict[tuple[str, ...], TorchFeedForwardBatch | None] = {}
        self._config = None

    @property
    def trainer_model_path(self) -> Path:
        runtime = get_runtime()
        return runtime.models_dir / runtime.game / "model.zip"

    @property
    def num_timesteps(self) -> int:
        return self.population_size * sum(self.generations_completed.values())

    def get_env(self):
        return self.env

    def learn(self, total_timesteps=None, callback=None, **kwargs):
        restart_attempted = bool(kwargs.pop("_restart_attempted", False))
        if total_timesteps is None:
            raise ValueError("total_timesteps is required.")
        if total_timesteps <= 0:
            return self

        generations_remaining = math.ceil(total_timesteps / self.population_size)
        total_generations = generations_remaining
        self.env.reset()
        num_inputs = len(self.env.env_method("policy_input")[0][0])
        num_outputs = self.env.env_method("num_actions")[0]
        self._ensure_input_compatibility(num_inputs)
        state_names = self._training_state_names()

        logger.info(
            "Generating NEAT config: "
            f"{num_inputs} inputs, {num_outputs} outputs, "
            f"{len(state_names)} state populations"
        )
        write_neat_config(
            path=self.config_path, num_inputs=num_inputs, num_outputs=num_outputs, pop_size=self.population_size
        )
        self._config = None
        config = self._load_config()
        training_callback = self._initialize_callback(callback)
        if training_callback is not None:
            training_callback.on_training_start(locals(), globals())

        try:
            continue_training = True
            for state_index, state_name in enumerate(state_names):
                if state_index < self._highest_progress_index(state_names):
                    logger.info(f"Skipping beaten training state {state_name}")
                    continue

                population = self._load_population(state_name, config)
                logger.info(
                    f"Training NEAT population for {state_name} "
                    f"from generation {population.generation} "
                    f"({generations_remaining} generations in timestep budget)"
                )
                self.populations[state_name] = population
                self.population = population
                self._add_reporters(population, state_name, total_generations)

                evaluator = NEATEvaluator(
                    self.env,
                    training_state=state_name,
                    controller_genomes=dict(self.winners),
                    max_steps=20_000,
                    max_no_progress_steps=600,
                    episodes_per_genome=3,
                    callback=training_callback,
                )
                while generations_remaining > 0:
                    if training_callback is not None:
                        training_callback.on_rollout_start()
                    winner = population.run(evaluator.evaluate_generation, 1)
                    generations_remaining -= 1
                    self.generations_completed[state_name] = population.generation
                    self.winners[state_name] = winner
                    self.winner = winner
                    self._networks.clear()
                    self._torch_network_batches.clear()
                    self._save_all()
                    if training_callback is not None:
                        training_callback.on_rollout_end()

                    if not evaluator.continue_training:
                        if evaluator.restart_requested and not restart_attempted:
                            logger.warning(
                                "Restarting NEAT training from scratch after incompatible controller genomes."
                            )
                            self._reset_training_state()
                            return self.learn(
                                total_timesteps=total_timesteps,
                                callback=callback,
                                _restart_attempted=True,
                                **kwargs,
                            )
                        continue_training = False
                        break

                    if self._highest_progress_index(state_names) > state_index:
                        logger.info(f"Advancing training beyond beaten state {state_name}")
                        break

                if generations_remaining == 0 or not continue_training:
                    break
        finally:
            if training_callback is not None:
                training_callback.on_training_end()

        self._save_all()
        return self

    def _reset_training_state(self) -> None:
        self.generations_completed.clear()
        self.winners.clear()
        self.winner = None
        self.populations.clear()
        self.population = None
        self.statistics.clear()
        self._networks.clear()
        self._torch_network_batches.clear()

    def _initialize_callback(self, callback) -> BaseCallback | None:
        if callback is None:
            return None
        if isinstance(callback, list):
            callback = CallbackList(callback)
        if not isinstance(callback, BaseCallback):
            raise TypeError("callback must be a BaseCallback or a list of BaseCallback instances.")
        callback.init_callback(self)
        return callback

    def _load_population(self, state_name: str, config) -> neat.Population:
        for checkpoint in self._checkpoints(state_name):
            logger.info(f"Restoring NEAT population from {checkpoint}")
            try:
                population = neat.Checkpointer.restore_checkpoint(
                    str(checkpoint),
                    new_config=config,
                )
            except Exception as error:
                logger.warning(f"Ignoring unreadable checkpoint {checkpoint}: {error}")
                continue

            self._synchronize_node_indexer(config, population.population.values())
            self.generations_completed[state_name] = population.generation
            return population

        return self._replacement_population(state_name, config)

    def _replacement_population(self, state_name: str, config) -> neat.Population:
        population = neat.Population(config)
        winner = self.winners.get(state_name)
        completed = self.generations_completed.get(state_name, 0)

        if winner is None or completed == 0:
            logger.info(f"No readable checkpoint found for {state_name}; creating a new population")
            return population

        self._synchronize_node_indexer(config, [winner])

        keys = list(population.population)
        tracker = population.reproduction.innovation_tracker
        tracker.global_counter = max(
            (connection.innovation for connection in winner.connections.values()),
            default=0,
        )
        tracker.generation_innovations.clear()
        config.genome_config.innovation_tracker = tracker

        population.population.clear()
        for index, key in enumerate(keys):
            genome = deepcopy(winner)
            genome.key = key
            if index > 0:
                genome.mutate(config.genome_config)
            population.population[key] = genome

        population.species = config.species_set_type(
            config.species_set_config,
            population.reporters,
        )
        population.species.speciate(
            config,
            population.population,
            completed,
        )
        population.generation = completed
        logger.info(
            f"No readable checkpoint found for {state_name}; "
            f"rebuilding generation {completed} from its saved winner"
        )
        return population

    @staticmethod
    def _synchronize_node_indexer(config, genomes) -> None:
        genome_config = config.genome_config
        next_node_key = genome_config.num_outputs

        if genome_config.node_indexer is not None:
            next_node_key = max(next_node_key, next(copy(genome_config.node_indexer)))

        for genome in genomes:
            if genome.nodes:
                next_node_key = max(next_node_key, max(genome.nodes) + 1)

        genome_config.node_indexer = count(next_node_key)

    def _checkpoints(self, state_name: str) -> list[Path]:
        state_dir = self.output_dir / state_name
        checkpoints = []
        for path in state_dir.glob("checkpoint-*"):
            try:
                generation = int(path.name.rsplit("-", 1)[1])
            except ValueError:
                continue
            checkpoints.append((generation, path))

        checkpoints.sort(reverse=True, key=lambda item: item[0])
        return [path for _, path in checkpoints]

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "winners": self.winners,
            "winner": self.winner,
            "generations_completed": self.generations_completed,
            "population_size": self.population_size,
            "input_signature": self.input_signature,
        }
        temporary_path = path.with_name(f".{path.name}.tmp")
        with temporary_path.open("wb") as file:
            pickle.dump(payload, file)
        temporary_path.replace(path)

    @classmethod
    def load(
        cls,
        path,
        env=None,
        settings_path: str | Path = DEFAULT_CONFIG_PATH,
        **kwargs,
    ):
        path = Path(path)
        zip_path = Path(f"{path}.zip")
        if not path.exists() and zip_path.exists():
            path = zip_path

        with path.open("rb") as file:
            payload = pickle.load(file)

        model = cls(
            env=env,
            generations_completed=payload["generations_completed"],
            winners=payload["winners"],
            winner=payload["winner"],
            input_signature=payload.get("input_signature"),
            settings_path=settings_path,
        )
        return model

    def predict(self, observation, deterministic=True):
        if self.env is None:
            raise RuntimeError("An environment is required to route state-specific winners.")

        config = self._load_config()
        policy_inputs = self.env.env_method("policy_input")
        all_features, state_names = zip(*policy_inputs, strict=True)
        if all_features:
            self._ensure_input_compatibility(len(all_features[0]))
        actions = [0] * len(state_names)
        routed_networks = []

        for index, (state_name, features) in enumerate(zip(state_names, all_features, strict=True)):
            genome = self.winners.get(state_name)
            if genome is None:
                continue

            if state_name not in self._networks:
                self._networks[state_name] = neat.nn.FeedForwardNetwork.create(
                    genome,
                    config,
                )
            routed_networks.append((index, state_name, features, self._networks[state_name]))

        batch_key = tuple(state_name for _, state_name, _, _ in routed_networks)
        if batch_key not in self._torch_network_batches:
            self._torch_network_batches[batch_key] = TorchFeedForwardBatch.create(
                [network for _, _, _, network in routed_networks]
            )

        torch_batch = self._torch_network_batches[batch_key]
        if torch_batch is not None:
            outputs = torch_batch.activate([features for _, _, features, _ in routed_networks])
            for (index, _, _, _), network_outputs in zip(routed_networks, outputs, strict=True):
                actions[index] = int(np.argmax(network_outputs))
        else:
            for index, _, features, network in routed_networks:
                actions[index] = int(np.argmax(network.activate(features)))

        return np.asarray(actions, dtype=np.int64), None

    def _training_state_names(self) -> list[str]:
        names = self.env.env_method("training_state_names")[0]
        if not names:
            raise ValueError("The environment did not declare any training states.")
        if len(names) != len(set(names)):
            raise ValueError(f"Training state names must be unique: {names}")
        return names

    def _highest_progress_index(self, state_names: list[str]) -> int:
        highest_states = self.env.env_method("highest_progress_state")
        return max(state_names.index(state_name) for state_name in highest_states)

    def _current_input_signature(self, num_inputs: int) -> dict[str, int | str]:
        encoder_source = inspect.getsource(FixedVisualEncoder.encode)
        encoder_kernels = np.asarray(FixedVisualEncoder._kernels, dtype=np.float32)
        encoder_hash = hashlib.sha256()
        encoder_hash.update(encoder_source.encode("utf-8"))
        encoder_hash.update(str(FixedVisualEncoder.output_size).encode("utf-8"))
        encoder_hash.update(str(FixedVisualEncoder._pooled_size).encode("utf-8"))
        encoder_hash.update(encoder_kernels.tobytes())
        return {
            "num_inputs": int(num_inputs),
            "encoder_signature": encoder_hash.hexdigest(),
        }

    def _ensure_input_compatibility(self, num_inputs: int) -> None:
        current_signature = self._current_input_signature(num_inputs)
        if not self.input_signature:
            self.input_signature = current_signature
            return

        if self.input_signature == current_signature:
            return

        expected_inputs = self.input_signature.get("num_inputs")
        expected_encoder_signature = self.input_signature.get("encoder_signature")
        logger.warning(
            "Model input has changed. "
            f"Expected {expected_inputs} inputs, got {current_signature['num_inputs']}. "
            f"Encoder signature changed: {expected_encoder_signature != current_signature['encoder_signature']}."
        )
        logger.warning("Deleting all model artifacts and stopping training.")
        settings.empty_all_paths()
        raise ValueError("Model input has changed. Model files were deleted. Please restart training.")

    def _load_config(self):
        if self._config is None:
            self._config = neat.Config(
                neat.DefaultGenome,
                neat.DefaultReproduction,
                neat.DefaultSpeciesSet,
                neat.DefaultStagnation,
                str(self.config_path),
            )
        return self._config

    def _add_reporters(
        self,
        population: neat.Population,
        state_name: str,
        total_generations: int,
    ) -> None:
        population.add_reporter(
            AdaptiveConfigReporter(
                population_size=self.population_size,
                total_generations=total_generations,
            )
        )
        population.add_reporter(LoguruReporter(show_species_detail=True))
        statistics = neat.StatisticsReporter()
        self.statistics[state_name] = statistics
        population.add_reporter(statistics)

        state_dir = self.output_dir / state_name
        state_dir.mkdir(parents=True, exist_ok=True)
        population.add_reporter(
            AtomicCheckpointer(
                generation_interval=1,
                filename_prefix=str(state_dir / "checkpoint-"),
            )
        )
        population.add_reporter(
            WinnerReporter(
                self,
                state_name,
                savestate_stagnation=max(5, min(20, math.ceil(total_generations * 0.1))),
            )
        )

    def delete_savestate(self, state_name: str) -> bool:
        deleted = self.env.env_method("delete_savestate", state_name)
        return any(deleted)

    def _save_all(self) -> None:
        self.save(self.output_dir / "winners.pkl")
        self.save(self.output_dir / "winner.pkl")
        self.save(self.trainer_model_path)
