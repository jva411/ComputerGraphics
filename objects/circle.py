import numba
import numpy as np
from utils.ray import Ray
from utils.material import BLANK
from objects.plane import Plane, t_correction


class Circle(Plane):
    def __init__(self, position: np.ndarray, normal: np.ndarray, radius: float, material = BLANK):
        super().__init__(position, normal, material)
        self.radius = radius

    def intersects(self, ray: Ray) -> np.ndarray:
        return intersects(ray, self.position, self.normal, self.radius)


@numba.jit(cache=True)
def intersects(ray, position, normal, radius):
    dn = ray.direction @ normal
    if dn == 0: return None

    t = (position - ray.origin) @ normal / dn - t_correction
    if t < 0 or ray.t < t: return None

    point = ray.origin + ray.direction * t
    distance = np.linalg.norm(point - position)
    if distance > radius: return None

    ray.t = t
    return point
