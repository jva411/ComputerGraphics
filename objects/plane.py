import numba
import numpy as np
from utils.ray import Ray
from utils import transforms
from utils.material import BLANK
from objects.object import Object, t_correction


class Plane(Object):
    def __init__(self, position: np.ndarray, normal: np.ndarray, material = BLANK):
        super().__init__(position, material)
        self.normal = transforms.normalize(normal)

        dirXZ = self.normal[[0, 2]]
        if all(dirXZ == np.array([0., 0.])):
            aXZ = 0
        else:
            rDirXZ = transforms.rotate2D(transforms.normalize(dirXZ), -np.pi/2)
            dX = rDirXZ @ np.array([1., 0.])
            dZ = rDirXZ @ np.array([0., 1.])
            aXZ = np.arccos(dX)
            if (dZ < 0): aXZ = 2*np.pi - aXZ

        self.__right = transforms.rotateY(np.array([1., 0., 0.]), -aXZ)
        self.__up = transforms.rotate(self.normal, -np.pi/2, self.__right)

    def intersects(self, ray: Ray) -> np.ndarray:
        return intersects(ray, self.position, self.normal)

    def getNormal(self, point: np.ndarray) -> np.ndarray:
        return self.normal

    def getColor(self, point: np.ndarray) -> np.ndarray:
        if self.material.texture is None:
            return self.material.color

        po = point - self.position
        pon = transforms.normalize(po)
        du = pon @ self.__up
        dr = pon @ self.__right
        angle = np.arccos(dr)
        if du < 0: angle = 2*np.pi - angle

        texture_point = transforms.rotate2D(np.array([1., 0.]), angle) * np.linalg.norm(po)
        return self.material.texture.getColor(texture_point)


@numba.jit(cache=True)
def intersects(ray, position, normal):
    dn = ray.direction @ normal
    if dn == 0: return None

    t = (position - ray.origin) @ normal / dn - t_correction
    if t < 0 or ray.t < t: return None

    ray.t = t
    return ray.origin + ray.direction * t
