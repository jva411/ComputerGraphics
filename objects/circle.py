import numpy as np
from utils.ray import Ray
from utils.material import BLANK
from utils.core import dynamic_lib
from objects.plane import Plane, t_correction
from ctypes import CDLL, c_void_p, c_double


intersects = dynamic_lib.circleIntersection
intersects.restype = c_double


class Circle(Plane):
    def __init__(self, position: np.ndarray, normal: np.ndarray, radius: float, material = BLANK):
        super().__init__(position, normal, material)
        self.radius = radius

        self.positionP, self.normalP, self.radiusC = None, None, None

    def preCalc(self, reverse=False):
        if reverse:
            self.positionP = None
            self.normalP = None
            self.radiusC = None
        else:
            self.positionP = c_void_p(self.position.ctypes.data)
            self.normalP = c_void_p(self.normal.ctypes.data)
            self.radiusC = c_double(self.radius)

    def intersects(self, ray: Ray) -> np.ndarray:
        t = intersects(ray.originP, ray.directionP, ray.tC, self.positionP, self.normalP, self.radiusC)
        if t>0:
            ray.t = t
            return ray.hitting_point
