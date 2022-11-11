import numpy as np
from numba import jit
from utils.ray import Ray
from utils import transforms
from utils.material import BLANK
from objects.object import Object, t_correction


class Sphere(Object):
    def __init__(self, position: np.ndarray, radius: float, material = BLANK):
        super().__init__(position, material)
        self.radius = radius


    def intersects(self, ray: Ray) -> np.ndarray:
        t, point = intersects(self.position, self.radius, ray.origin, ray.direction, ray.t)
        if t is not None:
            ray.t = t

        return point

    def getNormal(self, point: np.ndarray) -> np.ndarray:
        return (point - self.position) / self.radius

@jit(nopython=True)
def intersects(position, radius, rayOrigin, rayDirection, tMax):
    co = rayOrigin - position

    b = 2 * co @ rayDirection
    c = co @ co - radius ** 2
    delta = b ** 2 - 4*c
    if delta < 0: return None, None

    ts = []
    t1 = (-b + np.sqrt(delta)) / 2
    t2 = (-b - np.sqrt(delta)) / 2
    if 0 < t1 < tMax:
        ts.append(t1)
    if 0 < t2 < tMax:
        ts.append(t2)
    if len(ts) == 0:
        return None, None

    t = min(ts) - t_correction
    return t, rayOrigin + rayDirection*t
