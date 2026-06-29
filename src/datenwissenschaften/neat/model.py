import math
import pickle
from copy import copy, deepcopy
from itertools import count
from pathlib import Path

import numpy as np

import neat
from src.neat.checkpointer import AtomicCheckpointer
from src.neat.config import write_neat_config
from src.neat.evaluator import NEATEvaluator
from src.neat.reporter import AdaptiveConfigReporter, WinnerReporter


class NEATModel:
    trainer_model_path = Path("working/model/SnakeRattleNRoll-Nes-v0/model.zip")

    def __init__(
            self,
            env,
            config_path: Path,
            output_dir: Path,
            population_size: int = 100,
            generations_completed: dict[str, int] | None = None,
            winners: dict[str, object] | None = None,
            winner=None,
    ):
        self.env = env
        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir)
        self.population_size = population_size
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.generations_completed = dict(generations_completed or {})
        self.winners = dict(winners or {})
        self.winner = winner
        self.populations: dict[str, neat.Population] = {}
        self.population = None
        self.statistics: dict[str, neat.StatisticsReporter] = {}
        self._networks: dict[str, neat.nn.FeedForwardNetwork] = {}
        self._config = None

    @property
    def num_timesteps(self) -> int:
        return self.population_size * sum(self.generations_completed.values())

    def learn(self, total_timesteps=None, callback=None, **kwargs):
        if total_timesteps is None:
            raise ValueError("total_timesteps is required.")
        if total_timesteps <= 0:
            return self

        generations_remaining = math.ceil(total_timesteps / self.population_size)
        total_generations = generations_remaining
        self.env.reset()
        num_inputs = len(self.env.env_method("features")[0])
        num_outputs = self.env.env_method("num_actions")[0]
        state_names = self._training_state_names()

        print(
            "Generating NEAT config: " f"{num_inputs} inputs, {num_outputs} outputs, " f"{len(state_names)} state populations")
        write_neat_config(
            path=self.config_path,
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            pop_size=self.population_size
        )
        self._config = None
        config = self._load_config()

        for state_index, state_name in enumerate(state_names):
            if state_index < self._highest_progress_index(state_names):
                print(f"Skipping beaten training state {state_name}")
                continue

            population = self._load_population(state_name, config)
            print(
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
            )
            while generations_remaining > 0:
                winner = population.run(evaluator.evaluate_generation, 1)
                generations_remaining -= 1
                self.winners[state_name] = winner
                self.winner = winner
                self._networks.clear()
                self._save_all()

                if self._highest_progress_index(state_names) > state_index:
                    print(f"Advancing training beyond beaten state {state_name}")
                    break

            if generations_remaining == 0:
                break

        self._save_all()
        return self

    def _load_population(self, state_name: str, config) -> neat.Population:
        for checkpoint in self._checkpoints(state_name):
            print(f"Restoring NEAT population from {checkpoint}")
            try:
                population = neat.Checkpointer.restore_checkpoint(
                    str(checkpoint),
                    new_config=config,
                )
            except Exception as error:
                print(f"Ignoring unreadable checkpoint {checkpoint}: {error}")
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
            print(f"No readable checkpoint found for {state_name}; creating a new population")
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
        print(
            f"No readable checkpoint found for {state_name}; "
            f"rebuilding generation {completed} from its saved winner"
        )
        return population

    @staticmethod
    def _synchronize_node_indexer(config, genomes) -> None:
        """Keep NEAT's global node allocator ahead of all restored genomes."""
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
            "config_path": str(self.config_path),
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
            config_path: Path | None = None,
            output_dir: Path | None = None,
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
            config_path=config_path or Path(payload.get("config_path", "src/neat/config.ini")),
            output_dir=output_dir or Path("working/neat"),
            population_size=payload.get("population_size", 100),
            generations_completed=payload.get("generations_completed"),
            winners=payload.get("winners"),
            winner=payload.get("winner"),
        )
        if "population_size" not in payload:
            model.population_size = model._load_config().pop_size
        return model

    def predict(self, observation, deterministic=True):
        if self.env is None:
            raise RuntimeError("An environment is required to route state-specific winners.")

        config = self._load_config()
        state_names = self.env.env_method("state_name")
        all_features = self.env.env_method("features")
        winners = self._winners_with_legacy_fallback()
        actions = []

        for state_name, features in zip(state_names, all_features, strict=True):
            genome = winners.get(state_name)
            if genome is None:
                actions.append(0)
                continue

            if state_name not in self._networks:
                self._networks[state_name] = neat.nn.FeedForwardNetwork.create(
                    genome,
                    config,
                )
            outputs = self._networks[state_name].activate(features)
            actions.append(int(np.argmax(outputs)))

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

    def _winners_with_legacy_fallback(self) -> dict[str, object]:
        if self.winners or self.winner is None:
            return self.winners
        first_state = self._training_state_names()[0]
        return {first_state: self.winner}

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
        population.add_reporter(neat.StdOutReporter(True))
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
