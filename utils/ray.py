import numba
import numpy as np
from utils import transforms
from ctypes import c_void_p, c_double


class Ray:
    def __init__(self, origin: np.ndarray, direction: np.ndarray, t: float = np.inf):
        self.origin = origin
        self.direction = direction
        self.__t = t
        self.originP = c_void_p(self.origin.ctypes.data)
        self.directionP = c_void_p(self.direction.ctypes.data)
        self.tC = c_double(t)

    @property
    def t(self):
        return self.__t

    @t.setter
    def t(self, t):
        self.__t = t
        self.tC = c_double(t)

    @property
    def hitting_point(self):
        return self.origin + self.direction * self.t
