import numpy as np
from utils.ray import Ray
from utils.material import BLANK
from objects.plane import Plane, t_correction


class Circle(Plane):
    def __init__(self, position: np.ndarray, normal: np.ndarray, radius: float, material = BLANK):
        super().__init__(position, normal, material)
        self.radius = radius

    def intersects(self, ray: Ray) -> np.ndarray:
        dn = ray.direction @ self.normal
        if dn == 0: return None

        t = (self.position - ray.origin) @ self.normal / dn - t_correction
        if t < 0 or ray.t < t: return None

        distance = np.linalg.norm((ray.origin + ray.direction * t) - self.position)
        if distance > self.radius: return None

        ray.t = t
        return ray.hitting_point
