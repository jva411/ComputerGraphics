import numpy as np
from utils.ray import Ray
from utils import transforms
from objects.object import Object


class Plane(Object):
    def __init__(self, position: np.ndarray, normal: np.ndarray, color: np.ndarray = np.array([255., 255., 255.])):
        super().__init__(position, color)
        self.normal = transforms.normalize(normal)

    def intersects(self, ray: Ray) -> np.ndarray:
        dn = ray.direction @ self.normal
        if dn == 0: return None

        t = (self.position - ray.origin) @ self.normal / dn
        if t < 0 or ray.t < t: return None

        ray.t = t
        return ray.hitting_point

    def getNormal(self, point: np.ndarray) -> np.ndarray:
        return self.normal
