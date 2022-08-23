import numpy as np
from utils.ray import Ray
from utils import transforms
from objects.object import Object


class Sphere(Object):
    def __init__(self, position: np.ndarray, radius: float, color: np.ndarray = np.array([255., 255., 255.])):
        super().__init__(position, color)
        self.radius = radius


    def intersects(self, ray: Ray) -> np.ndarray:
        m: np.ndarray = ray.origin - self.position
        b = m @ ray.direction
        c = m @ m - self.radius ** 2
        if c > 0 and b > 0: return None

        delta = b ** 2 - c
        if delta < 0: return None

        distance = -b - np.sqrt(delta)
        if distance < 0 or ray.t < distance: return None

        ray.t = distance
        return ray.hitting_point

    def getNormal(self, point: np.ndarray) -> np.ndarray:
        return transforms.normalize(point - self.position)
