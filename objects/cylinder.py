import math
import numpy as np
from utils.ray import Ray
from utils import transforms
from utils.material import BLANK
from utils.core import dynamic_lib
from objects.object import Object, t_correction
from ctypes import CDLL, c_void_p, c_double


intersects = dynamic_lib.cylinderIntersection
intersects.restype = c_double


class Cylinder(Object):
    def __init__(self, position: np.ndarray, axis: np.ndarray, height: float, radius: float, material = BLANK, center_top: np.array = None):
        super().__init__(position, material)
        self.axis = transforms.normalize(axis if center_top is None else (position - center_top))
        self.height = height if center_top is None else np.linalg.norm(center_top - position)
        self.radius = radius

        self.positionP, self.axisP, self.heightC, self.radiusC = None, None, None, None

        dirXZ = self.axis[[0, 2]]
        if all(dirXZ == np.array([0., 0.])):
            aXZ = 0
        else:
            rDirXZ = transforms.rotate2D(transforms.normalize(dirXZ), -np.pi/2)
            dX = rDirXZ @ np.array([1., 0.])
            dZ = rDirXZ @ np.array([0., 1.])
            aXZ = np.arccos(dX)
            if (dZ < 0): aXZ = 2*np.pi - aXZ

        self.__right = transforms.rotateY(np.array([1., 0., 0.]), -aXZ)
        self.__up = transforms.rotate(self.axis, -np.pi/2, self.__right)

    def preCalc(self, reverse=False):
        if reverse:
            self.positionP = None
            self.axisP = None
            self.heightC = None
            self.radiusC = None
        else:
            self.positionP = c_void_p(self.position.ctypes.data)
            self.axisP = c_void_p(self.axis.ctypes.data)
            self.heightC = c_double(self.height)
            self.radiusC = c_double(self.radius)


    def intersects(self, ray: Ray) -> np.ndarray:
        t = intersects(ray.originP, ray.directionP, ray.tC, self.positionP, self.axisP, self.radiusC, self.heightC)
        if t>0:
            ray.t = t
            return ray.hitting_point

    def getNormal(self, point: np.ndarray) -> np.ndarray:
        w = point - self.position
        n = w - self.axis * (w @ self.axis)
        return transforms.normalize(n)

    def getColor(self, point: np.ndarray):
        if self.material.texture is None:
            return self.material.color

        po = point - self.position
        n = self.getNormal(point)
        du = n @ self.__up
        dr = n @ self.__right
        angle = np.arccos(dr)
        if du < 0: angle = 2*np.pi - angle

        u = angle / (2*np.pi)
        v = (po @ self.axis) / self.height
        return self.material.texture.getColor(np.array([u, v]))
