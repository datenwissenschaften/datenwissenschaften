import math

import neat


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
                print(
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
