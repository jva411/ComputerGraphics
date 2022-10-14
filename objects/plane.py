import numpy as np
from utils.ray import Ray
from utils import transforms
from utils.material import BLANK
from objects.object import Object, t_correction


class Plane(Object):
    def __init__(self, position: np.ndarray, normal: np.ndarray, material = BLANK):
        super().__init__(position, material)
        self.normal = transforms.normalize(normal)
        # cosN = self.normal @ np.array([0., 0., -1.])
        # self.direction = transforms.rotateX(transforms.normalize(direction), np.arccos(cosN))

    def intersects(self, ray: Ray) -> np.ndarray:
        dn = ray.direction @ self.normal
        if dn == 0: return None

        t = (self.position - ray.origin) @ self.normal / dn - t_correction
        if t < 0 or ray.t < t: return None

        ray.t = t
        return ray.hitting_point

    def getNormal(self, point: np.ndarray) -> np.ndarray:
        return self.normal

    def getColor(self, point: np.ndarray) -> np.ndarray:
        if self.material.texture is None:
            return self.material.color

        # po = point - self.position
        # pon = transforms.normalize(po)
        # d = pon @ self.direction
        # c = np.cross(pon, self.direction)
        # dn = self.normal @ c
        # if dn < 0: d = -d
        # angle = np.arccos(d)

        # vPixel = transforms.rotate(np.arrayZ([1., 0., 0.], angle)) * np.linalg.norm(po)
