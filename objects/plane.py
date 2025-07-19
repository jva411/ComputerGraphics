import numpy as np
from utils.ray import Ray
from utils import transforms
from utils.material import BLANK
from utils.core import dynamic_lib
from objects.object import Object, t_correction
from ctypes import CDLL, c_void_p, c_double


intersects = dynamic_lib.planeIntersection
intersects.restype = c_double


class Plane(Object):
    def __init__(self, position: np.ndarray, normal: np.ndarray, material = BLANK):
        super().__init__(position, material)
        self.normal = transforms.normalize(normal)

        self.positionP, self.normalP = None, None

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

    def preCalc(self, reverse=False):
        if reverse:
            self.positionP = None
            self.normalP = None
        else:
            self.positionP = c_void_p(self.position.ctypes.data)
            self.normalP = c_void_p(self.normal.ctypes.data)

    def intersects(self, ray: Ray) -> np.ndarray:
        t = intersects(ray.originP, ray.directionP, ray.tC, self.positionP, self.normalP)
        if t>0:
            ray.t = t
            return ray.hitting_point

    def getNormal(self, point: np.ndarray) -> np.ndarray:
        return self.normal

    def getColor(self, point: np.ndarray) -> np.ndarray:
        if self.material.texture is None:
            return self.material.getColor()

        po = point - self.position
        pon = transforms.normalize(po)
        du = pon @ self.__up
        dr = pon @ self.__right
        angle = np.arccos(dr)
        if du < 0: angle = 2*np.pi - angle

        texture_point = transforms.rotate2D(np.array([1., 0.]), angle) * np.linalg.norm(po)
        return self.material.texture.getColor(texture_point)
