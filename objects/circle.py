import numpy as np
from utils.ray import Ray
from objects.plane import Plane


class Circle(Plane):
    def __init__(self, position: np.ndarray, normal: np.ndarray, radius: float, color: np.ndarray = np.array([255., 255., 255.])):
        super().__init__(position, normal, color)
        self.radius = radius
        print(self.position, self.normal, self.radius)

    def intersects(self, ray: Ray) -> np.ndarray:
        dn = ray.direction @ self.normal
        if dn == 0: return None

        t = (self.position - ray.origin) @ self.normal / dn
        if t < 0 or ray.t < t: return None

        distance = np.linalg.norm((ray.origin + ray.direction * t) - self.position)
        if distance > self.radius: return None

        ray.t = t
        return ray.hitting_point
