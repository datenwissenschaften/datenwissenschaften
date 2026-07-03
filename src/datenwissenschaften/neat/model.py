import hashlib
import inspect
import math
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
from datenwissenschaften.neat.policy import NEATPolicyRouter
from datenwissenschaften.neat.population import load_population, preserve_global_champion
from datenwissenschaften.neat.reporter import AdaptiveConfigReporter, LoguruReporter, WinnerReporter
from datenwissenschaften.neat.serialization import load_model_state, save_model_state
from datenwissenschaften.runtime import get_runtime
from datenwissenschaften.settings import DEFAULT_CONFIG_PATH, load_config
from datenwissenschaften.ui.control import consume_model_reset, perform_model_reset
from datenwissenschaften.ui.telemetry import publish_generation, publish_metadata
from datenwissenschaften.vision.encoder import FixedVisualEncoder


class NEATModel:
    supports_ui_restart = True

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

        self.best_fitness = float("-inf")
        self.best_genome = None
        self.generations_completed = dict(generations_completed or {})
        self.winners = dict(winners or {})
        self.winner = winner
        self.input_signature = dict(input_signature or {})
        self.populations: dict[str, neat.Population] = {}
        self.population = None
        self.statistics: dict[str, neat.StatisticsReporter] = {}
        self._policy_router = NEATPolicyRouter(self.winners)
        # Reporter compatibility: WinnerReporter invalidates state networks
        # through this existing private attribute.
        self._networks = self._policy_router.networks
        self._torch_network_batches = self._policy_router.torch_batches
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
        restart_budget = total_timesteps + self.num_timesteps if total_timesteps is not None else None
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
        self._publish_neat_metadata(config, num_inputs, num_outputs, state_names)
        training_callback = self._initialize_callback(callback)
        model_reset = None
        if training_callback is not None:
            training_callback.on_training_start(locals(), globals())

        try:
            continue_training = True
            while generations_remaining > 0 and continue_training:
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
                        callback=training_callback,
                    )
                    while generations_remaining > 0:
                        if training_callback is not None:
                            training_callback.on_rollout_start()
                        winner = population.run(evaluator.evaluate_generation, 1)
                        model_reset = consume_model_reset()
                        if model_reset is not None:
                            logger.warning("Stopping current NEAT generation for a requested model reset")
                            continue_training = False
                            break
                        self._preserve_global_champion(population, winner)
                        generations_remaining -= 1
                        self.generations_completed[state_name] = population.generation
                        publish_generation(
                            training_state=state_name,
                            generation=population.generation,
                            winner_fitness=float(getattr(winner, "fitness", 0.0) or 0.0),
                            species=len(population.species.species),
                        )
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

                        next_state_index = self._highest_progress_index(state_names)
                        if next_state_index != state_index:
                            if next_state_index == 0:
                                logger.info("Returning to initial-state training with no active automatic savestate")
                            else:
                                logger.info(f"Advancing training beyond beaten state {state_name}")
                            break

                    if generations_remaining == 0 or not continue_training:
                        break
        finally:
            if training_callback is not None:
                training_callback.on_training_end()

        model_reset = model_reset or consume_model_reset()
        if model_reset is not None:
            self._reset_training_state()
            perform_model_reset(model_reset)
            return self.learn(total_timesteps=restart_budget, callback=callback, reset_num_timesteps=True)

        self._save_all()
        return self

    def _publish_neat_metadata(self, config, num_inputs: int, num_outputs: int, state_names: list[str]) -> None:
        genome = config.genome_config
        publish_metadata(
            "neat",
            {
                "population_size": self.population_size,
                "inputs": num_inputs,
                "outputs": num_outputs,
                "hidden_nodes": genome.num_hidden,
                "training_states": state_names,
                "fitness_criterion": config.fitness_criterion,
                "fitness_threshold": config.fitness_threshold,
                "activation": genome.activation_default,
                "aggregation": genome.aggregation_default,
                "initial_connection": genome.initial_connection,
                "compatibility_threshold": config.species_set_config.compatibility_threshold,
                "max_stagnation": config.stagnation_config.max_stagnation,
                "elitism": config.reproduction_config.elitism,
                "survival_threshold": config.reproduction_config.survival_threshold,
                "node_add_probability": genome.node_add_prob,
                "node_delete_probability": genome.node_delete_prob,
                "connection_add_probability": genome.conn_add_prob,
                "connection_delete_probability": genome.conn_delete_prob,
            },
            replace=True,
        )

    def _reset_training_state(self) -> None:
        self.best_fitness = float("-inf")
        self.best_genome = None
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
        population = load_population(
            output_dir=self.output_dir,
            state_name=state_name,
            config=config,
            winners=self.winners,
            generations_completed=self.generations_completed,
        )
        if self.best_genome is not None:
            self._preserve_global_champion(population, self.best_genome)
        return population

    def save(self, path) -> None:
        payload = {
            "winners": self.winners,
            "winner": self.winner,
            "generations_completed": self.generations_completed,
            "population_size": self.population_size,
            "input_signature": self.input_signature,
            "best_fitness": self.best_fitness,
            "best_genome": self.best_genome,
        }
        save_model_state(path, payload)

    @classmethod
    def load(
        cls,
        path,
        env=None,
        settings_path: str | Path = DEFAULT_CONFIG_PATH,
        **kwargs,
    ):
        payload = load_model_state(path)

        model = cls(
            env=env,
            generations_completed=payload["generations_completed"],
            winners=payload["winners"],
            winner=payload["winner"],
            input_signature=payload.get("input_signature"),
            settings_path=settings_path,
        )

        saved_population_size = int(payload["population_size"])
        if saved_population_size != model.population_size:
            logger.warning(
                "NEAT population size changed "
                f"from {saved_population_size} to {model.population_size}. "
                "Deleting incompatible model artifacts and restarting training."
            )
            settings.empty_all_paths(settings_path)
            raise ValueError(f"NEAT population size changed from {saved_population_size} to {model.population_size}.")

        model.best_genome = payload.get("best_genome")
        if model.best_genome is None:
            candidates = [genome for genome in [*model.winners.values(), model.winner] if genome is not None]
            model.best_genome = max(
                candidates,
                key=lambda genome: genome.fitness if genome.fitness is not None else float("-inf"),
                default=None,
            )
        default_best_fitness = (
            float(model.best_genome.fitness)
            if model.best_genome is not None and model.best_genome.fitness is not None
            else float("-inf")
        )
        model.best_fitness = float(payload.get("best_fitness", default_best_fitness))

        return model

    def predict(self, observation, deterministic=True):
        if self.env is None:
            raise RuntimeError("An environment is required to route state-specific winners.")
        return self._policy_router.predict(
            env=self.env,
            config=self._load_config(),
            validate_input_size=self._ensure_input_compatibility,
        )

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

    def _preserve_global_champion(self, population: neat.Population, winner) -> None:
        self.best_fitness, self.best_genome = preserve_global_champion(
            population,
            winner,
            best_fitness=self.best_fitness,
            best_genome=self.best_genome,
        )
