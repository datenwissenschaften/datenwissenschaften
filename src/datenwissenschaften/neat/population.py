from __future__ import annotations

from collections.abc import Iterable
from copy import copy, deepcopy
from itertools import count
from pathlib import Path
from typing import Any

import neat
from loguru import logger


def load_population(
    *,
    output_dir: Path,
    state_name: str,
    config: Any,
    winners: dict[str, object],
    generations_completed: dict[str, int],
) -> neat.Population:
    for checkpoint in checkpoint_paths(output_dir, state_name):
        logger.info(f"Restoring NEAT population from {checkpoint}")
        try:
            population = neat.Checkpointer.restore_checkpoint(str(checkpoint), new_config=config)
        except Exception as error:
            logger.warning(f"Ignoring unreadable checkpoint {checkpoint}: {error}")
            continue

        synchronize_node_indexer(config, population.population.values())
        generations_completed[state_name] = population.generation
        return population

    return rebuild_population(
        state_name=state_name,
        config=config,
        winner=winners.get(state_name),
        completed=generations_completed.get(state_name, 0),
    )


def rebuild_population(*, state_name: str, config: Any, winner: Any, completed: int) -> neat.Population:
    population = neat.Population(config)
    if winner is None or completed == 0:
        logger.info(f"No readable checkpoint found for {state_name}; creating a new population")
        return population

    synchronize_node_indexer(config, [winner])
    tracker = population.reproduction.innovation_tracker
    tracker.global_counter = max(
        (connection.innovation for connection in winner.connections.values()),
        default=0,
    )
    tracker.generation_innovations.clear()
    config.genome_config.innovation_tracker = tracker

    keys = list(population.population)
    population.population.clear()
    for index, key in enumerate(keys):
        genome = deepcopy(winner)
        genome.key = key
        if index > 0:
            genome.mutate(config.genome_config)
        population.population[key] = genome

    population.species = config.species_set_type(config.species_set_config, population.reporters)
    population.species.speciate(config, population.population, completed)
    population.generation = completed
    logger.info(
        f"No readable checkpoint found for {state_name}; " f"rebuilding generation {completed} from its saved winner"
    )
    return population


def synchronize_node_indexer(config: Any, genomes: Iterable[Any]) -> None:
    genome_config = config.genome_config
    next_node_key = genome_config.num_outputs
    if genome_config.node_indexer is not None:
        next_node_key = max(next_node_key, next(copy(genome_config.node_indexer)))

    for genome in genomes:
        if genome.nodes:
            next_node_key = max(next_node_key, max(genome.nodes) + 1)
    genome_config.node_indexer = count(next_node_key)


def checkpoint_paths(output_dir: Path, state_name: str) -> list[Path]:
    checkpoints = []
    for path in (output_dir / state_name).glob("checkpoint-*"):
        try:
            generation = int(path.name.rsplit("-", 1)[1])
        except ValueError:
            continue
        checkpoints.append((generation, path))
    checkpoints.sort(reverse=True, key=lambda item: item[0])
    return [path for _, path in checkpoints]


def preserve_global_champion(
    population: neat.Population,
    winner: Any,
    *,
    best_fitness: float,
    best_genome: Any,
) -> tuple[float, Any]:
    """Track the all-time champion and ensure it survives reproduction.

    ``Population.run(..., 1)`` returns after reproduction and speciation. New
    offspring therefore have no fitness and cannot be ranked as "worst". We
    replace the newest unevaluated offspring, falling back to the lowest-fitness
    elite only when a generation consists entirely of elites. Speciation is
    repeated because species membership contains genome objects, not only keys.
    """
    winner_fitness = getattr(winner, "fitness", None)
    if winner_fitness is not None and float(winner_fitness) > best_fitness:
        best_fitness = float(winner_fitness)
        best_genome = deepcopy(winner)

    if best_genome is None or not population.population:
        return best_fitness, best_genome
    if _contains_genome(population, best_genome):
        return best_fitness, best_genome

    replacement_key = _replacement_key(population.population)
    source_key = best_genome.key
    champion = deepcopy(best_genome)
    champion.key = replacement_key
    champion.fitness = None
    population.population[replacement_key] = champion

    ancestors = getattr(population.reproduction, "ancestors", None)
    if isinstance(ancestors, dict):
        ancestors[replacement_key] = (source_key,)

    # Population.run increments generation after speciation. Match the
    # generation number used by the original speciation pass.
    species_generation = max(0, population.generation - 1)
    population.species.speciate(population.config, population.population, species_generation)
    logger.debug(f"Preserved global NEAT champion with fitness {best_fitness:.5f} " f"as genome {replacement_key}")
    return best_fitness, best_genome


def _contains_genome(population: neat.Population, genome: Any) -> bool:
    same_key = population.population.get(genome.key)
    candidates = ([same_key] if same_key is not None else []) + [
        candidate for key, candidate in population.population.items() if key != genome.key
    ]
    for candidate in candidates:
        try:
            if candidate.distance(genome, population.config.genome_config) == 0.0:
                return True
        except (AttributeError, TypeError, ValueError):
            if candidate is genome:
                return True
    return False


def _replacement_key(population: dict[int, Any]) -> int:
    unevaluated_keys = [key for key, genome in population.items() if genome.fitness is None]
    if unevaluated_keys:
        # neat-python assigns monotonically increasing keys to offspring, so
        # this avoids replacing an elite retained with its original key.
        return max(unevaluated_keys)
    return min(population, key=lambda key: (population[key].fitness, key))
