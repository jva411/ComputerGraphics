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
        return intersects(ray, self.position, self.radius)

    def getNormal(self, point: np.ndarray) -> np.ndarray:
        return (point - self.position) / self.radius

@jit
def intersects(ray, position, radius):
    co = ray.origin - position

    b = co @ ray.direction
    c = co @ co - radius ** 2
    delta = b ** 2 - c
    if delta < 0: return None

    delta2 = delta**0.5

    t1 = (-b + delta2) / 2
    t2 = (-b - delta2) / 2
    t = ray.t
    if 0 < t1 < t: t = t1
    if 0 < t2 < t: t = t2
    if t == ray.t: return None

    ray.t = t
    return ray.origin + ray.direction*t
