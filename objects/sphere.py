import numpy as np
from utils.ray import Ray
from utils import transforms
from utils.material import BLANK
from objects.object import Object, t_correction


class Sphere(Object):
    def __init__(self, position: np.ndarray, radius: float, material = BLANK):
        super().__init__(position, material)
        self.radius = radius


    def intersects(self, ray: Ray) -> np.ndarray:
        co = ray.origin - self.position

        b = 2 * co @ ray.direction
        c = co @ co - self.radius ** 2
        delta = b ** 2 - 4*c
        if delta < 0: return None

        ts = []
        t1 = (-b + np.sqrt(delta)) / 2
        t2 = (-b - np.sqrt(delta)) / 2
        if 0 < t1 < ray.t:
            ts.append(t1)
        if 0 < t2 < ray.t:
            ts.append(t2)
        if len(ts) == 0:
            return None

        t = min(ts) - t_correction
        ray.t = t
        return ray.hitting_point

    def getNormal(self, point: np.ndarray) -> np.ndarray:
        return (point - self.position) / self.radius
