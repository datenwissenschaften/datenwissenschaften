import math
import time
from statistics import mean, stdev

import neat
from loguru import logger


class AdaptiveConfigReporter(neat.reporting.BaseReporter):
    def __init__(self, *, population_size: int, total_generations: int):
        self.target_species = max(1, min(population_size, round(math.sqrt(population_size))))
        self.base_stagnation = max(5, min(20, math.ceil(total_generations * 0.2)))
        self.maximum_stagnation = max(self.base_stagnation, math.ceil(total_generations * 0.5))

    def post_evaluate(
        self,
        config,
        population,
        species,
        best_genome,
    ) -> None:
        species_count = max(1, len(species.species))
        average_species_size = len(population) / species_count

        self._adjust_compatibility_threshold(config, species_count)
        self._adjust_stagnation(config, species_count)
        self._adjust_reproduction(config, average_species_size, species_count, len(population))

    def _adjust_compatibility_threshold(self, config, species_count: int) -> None:
        difference = species_count - self.target_species
        adjustment = min(0.5, abs(difference) * 0.1)
        threshold = config.species_set_config.compatibility_threshold

        if difference > 0:
            threshold += adjustment
        elif difference < 0:
            threshold -= adjustment

        config.species_set_config.compatibility_threshold = min(10.0, max(0.5, threshold))

    def _adjust_stagnation(self, config, species_count: int) -> None:
        diversity_ratio = species_count / self.target_species
        patience = round(self.base_stagnation / math.sqrt(diversity_ratio))
        config.stagnation_config.max_stagnation = min(
            self.maximum_stagnation,
            max(5, patience),
        )
        config.stagnation_config.species_elitism = min(
            species_count,
            max(1, round(self.target_species * 0.2)),
        )

    @staticmethod
    def _adjust_reproduction(config, average_species_size: float, species_count: int, population_size: int) -> None:
        elitism = 1 if average_species_size < 10 else 2
        maximum_species_size = max(1, population_size // species_count)
        elitism = min(elitism, maximum_species_size)
        config.reproduction_config.elitism = elitism
        config.reproduction_config.min_species_size = elitism

        # Keep at least two expected parents per average species without
        # weakening selection beyond the upper half of that species.
        config.reproduction_config.survival_threshold = min(
            0.5,
            max(0.2, 2.0 / average_species_size),
        )


class WinnerReporter(neat.reporting.BaseReporter):
    def __init__(self, model, state_name: str, *, savestate_stagnation: int):
        self.model = model
        self.state_name = state_name
        self.savestate_stagnation = savestate_stagnation
        self.best_fitness = float("-inf")
        self.stagnant_generations = 0

    def post_evaluate(
        self,
        config,
        population,
        species,
        best_genome,
    ):
        fitness = float(best_genome.fitness)
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.stagnant_generations = 0
        else:
            self.stagnant_generations += 1

        if self.stagnant_generations >= self.savestate_stagnation:
            if self.model.delete_savestate(self.state_name):
                logger.info(
                    f"Deleted stagnant savestate for {self.state_name} "
                    f"after {self.stagnant_generations} generations without improvement"
                )
            self.best_fitness = float("-inf")
            self.stagnant_generations = 0

        self.model.winners[self.state_name] = best_genome
        self.model.winner = best_genome
        self.model.generations_completed[self.state_name] = self.model.population.generation + 1
        self.model._networks.pop(self.state_name, None)
        self.model._save_all()


class LoguruReporter(neat.reporting.BaseReporter):
    def __init__(self, show_species_detail: bool = True):
        self.show_species_detail = show_species_detail
        self.generation = None
        self.generation_start_time = None
        self.generation_times = []
        self.num_extinctions = 0

    def start_generation(self, generation):
        self.generation = generation
        logger.info(f"Running generation {generation}")
        self.generation_start_time = time.time()

    def end_generation(self, config, population, species_set):
        ng = len(population)
        ns = len(species_set.species)
        if self.show_species_detail:
            logger.info(f"Population of {ng:d} members in {ns:d} species (after reproduction)")
            for sid in sorted(species_set.species):
                species = species_set.species[sid]
                age = self.generation - species.created
                size = len(species.members)
                fitness = "--" if species.fitness is None else f"{species.fitness:.3f}"
                adjusted_fitness = "--" if species.adjusted_fitness is None else f"{species.adjusted_fitness:.3f}"
                stagnation = self.generation - species.last_improved
                logger.info(
                    f"species id={sid} age={age} size={size} "
                    f"fitness={fitness} adj_fit={adjusted_fitness} stag={stagnation}"
                )
        else:
            logger.info(f"Population of {ng:d} members in {ns:d} species (after reproduction)")

        elapsed = time.time() - self.generation_start_time
        self.generation_times.append(elapsed)
        self.generation_times = self.generation_times[-10:]
        average = sum(self.generation_times) / len(self.generation_times)

        logger.info(f"Total extinctions: {self.num_extinctions:d}")
        if len(self.generation_times) > 1:
            logger.debug(f"Generation time: {elapsed:.3f} sec ({average:.3f} average)")
        else:
            logger.debug(f"Generation time: {elapsed:.3f} sec")

    def post_evaluate(self, config, population, species, best_genome):
        fitnesses = [candidate.fitness for candidate in population.values()]
        fit_mean = mean(fitnesses)
        fit_std = stdev(fitnesses)
        best_species_id = species.get_species_id(best_genome.key)
        logger.info(f"Population average fitness: {fit_mean:3.5f} stdev: {fit_std:3.5f}")
        logger.info(
            "Best fitness: "
            f"{best_genome.fitness:3.5f} - size: {best_genome.size()!r} "
            f"- species {best_species_id} - id {best_genome.key}"
        )

    def complete_extinction(self):
        self.num_extinctions += 1
        logger.warning("All species extinct.")

    def found_solution(self, config, generation, best):
        logger.info(
            f"Best individual in generation {self.generation} meets fitness threshold " f"- complexity: {best.size()!r}"
        )

    def species_stagnant(self, sid, species):
        if self.show_species_detail:
            logger.info(f"Species {sid} with {len(species.members)} members is stagnated: removing it")

    def info(self, msg):
        logger.info(msg)
