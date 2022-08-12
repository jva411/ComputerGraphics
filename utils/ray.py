import numpy as np
from utils import transforms


class Ray:
    def __init__(self, origin: np.ndarray, direction: np.ndarray, t: float = np.inf):
        self.origin = origin
        self.direction = transforms.normalize(direction)
        self.t = t

    @property
    def hitting_point(self) -> np.ndarray:
        return self.origin + self.direction * self.t
