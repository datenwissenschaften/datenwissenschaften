import gzip
import pickle
import random
from pathlib import Path

import neat


class AtomicCheckpointer(neat.Checkpointer):
    """A NEAT checkpointer that cannot expose a partially written checkpoint."""

    def save_checkpoint(self, config, population, species_set, generation) -> None:
        path = Path(f"{self.filename_prefix}{generation}")
        temporary_path = path.with_name(f".{path.name}.tmp")
        print(f"Saving checkpoint to {path}")

        # Species reporters point back to the model and its multiprocessing
        # environment. They are runtime state and are rebuilt by Population.
        reporters = species_set.reporters
        species_set.reporters = neat.reporting.ReporterSet()
        try:
            with gzip.open(temporary_path, "wb", compresslevel=5) as file:
                pickle.dump(
                    (generation, config, population, species_set, random.getstate()),
                    file,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
        finally:
            species_set.reporters = reporters

        temporary_path.replace(path)
