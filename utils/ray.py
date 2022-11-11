import numba
import numpy as np
from utils import transforms


@numba.experimental.jitclass([
    ('origin', numba.float64[::1]),
    ('direction', numba.float64[::1]),
    ('t', numba.float64)
])
class Ray:
    def __init__(self, origin: np.ndarray, direction: np.ndarray, t: float = np.inf):
        self.origin = origin
        self.direction = direction
        self.t = t
