import numpy as np
from utils.ray import Ray
from utils import transforms
from utils.material import BLANK
from utils.core import dynamic_lib
from objects.object import Object, t_correction
from ctypes import CDLL, c_void_p, c_double


intersects = dynamic_lib.sphereIntersection
intersects.restype = c_double


class Sphere(Object):
    def __init__(self, position: np.ndarray, radius: float, material = BLANK):
        super().__init__(position, material)
        self.radius = radius
        self.positionP = None
        self.radiusC = None

    def preCalc(self, reverse=False):
        if reverse:
            self.positionP = None
            self.radiusC = None
        else:
            self.positionP = c_void_p(self.position.ctypes.data)
            self.radiusC = c_double(self.radius)

    def intersects(self, ray: Ray) -> np.ndarray:
        t = intersects(ray.originP, ray.directionP, ray.tC, self.positionP, self.radiusC)
        if t>0:
            ray.t = t
            return ray.hitting_point

    def getNormal(self, point: np.ndarray) -> np.ndarray:
        return (point - self.position) / self.radius
