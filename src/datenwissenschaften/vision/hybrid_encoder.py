import numpy as np

from datenwissenschaften.ram import RamInfo
from datenwissenschaften.vision.encoder import FixedVisualEncoder


class HybridEncoder:
    def __init__(self):
        self.visual_encoder = FixedVisualEncoder()

    def encode(self, observation: np.ndarray, ram: RamInfo) -> list[float]:
        return self.visual_encoder.encode(observation) + ram.features()
