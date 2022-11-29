import numpy as np
from numba import jit
from utils.ray import Ray
from utils import transforms
from utils.material import BLANK
from objects.object import Object, t_correction
from ctypes import CDLL, c_void_p, c_double

lib = CDLL('.\\utils\\core.so')
intersects = lib.sphereIntersection
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
        # t = intersects(ray.origin, ray.direction, ray.t, self.position, self.radius)
        t = intersects(ray.originP, ray.directionP, ray.tC, self.positionP, self.radiusC)
        if t>0:
            ray.t = t
            return ray.hitting_point

    def getNormal(self, point: np.ndarray) -> np.ndarray:
        return (point - self.position) / self.radius

# @jit
# def intersects(rayOrigin, rayDirection, rayT, position, radius):
#     co = rayOrigin - position

#     b = 2*co @ rayDirection
#     c = co @ co - radius ** 2
#     delta = b ** 2 - 4*c
#     if delta < 0: return -1.

#     delta2 = delta**0.5

#     t1 = (-b + delta2) / 2
#     t2 = (-b - delta2) / 2
#     t = rayT
#     if 0 < t1 < t: t = t1
#     if 0 < t2 < t: t = t2
#     if t == rayT: return -1.

#     return t
