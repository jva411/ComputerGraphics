import numpy as np
from utils.ray import Ray
from utils import transforms
from objects.object import Object


class Sphere(Object):
    def __init__(self, position: np.ndarray, radius: float, color: np.ndarray = np.array([255., 255., 255.])):
        super().__init__(position, color)
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

        t = min(ts)
        ray.t = t
        return ray.hitting_point

    def getNormal(self, point: np.ndarray) -> np.ndarray:
        return transforms.normalize(point - self.position)
