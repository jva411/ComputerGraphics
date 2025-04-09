import math
import numpy as np
from utils.ray import Ray
from utils import transforms
from utils.material import BLANK
from utils.core import dynamic_lib
from objects.object import Object, t_correction
from ctypes import CDLL, c_void_p, c_double


intersects = dynamic_lib.coneIntersection
intersects.restype = c_double


class Cone(Object):
    def __init__(self, position: np.ndarray, axis: np.ndarray, height: float, radius: float, material=BLANK, vertice: np.ndarray = None):
        super().__init__(position, material)
        self.axis = transforms.normalize(axis if not vertice else (vertice - position))
        self.height = height
        self.radius = radius
        self.__hip = math.sqrt(height ** 2 + radius ** 2)
        self.__cos = height / self.__hip
        self.__cos2 = self.__cos ** 2

        self.positionP, self.axisP, self.heightC, self.__cos2C = None, None, None, None

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
            self.__cos2C = None
        else:
            self.positionP = c_void_p(self.position.ctypes.data)
            self.axisP = c_void_p(self.axis.ctypes.data)
            self.heightC = c_double(self.height)
            self.__cos2C = c_double(self.__cos2)

    def intersects(self, ray: Ray) -> np.ndarray:
        t = intersects(ray.originP, ray.directionP, ray.tC, self.positionP, self.axisP, self.__cos2C, self.heightC)
        if t>0:
            ray.t = t
            return ray.hitting_point

    def getNormal(self, point: np.ndarray) -> np.ndarray:
        v = point - self.position
        n = v - self.axis * (self.axis @ v)
        return transforms.normalize(n)

    def rotateX(self, angle: float, around: np.ndarray = None):
        if around is not None:
            self.position = around + transforms.rotateX(self.position - around, angle)
        else:
            self.axis = transforms.rotateX(self.axis, angle)
    def rotateY(self, angle: float, around: np.ndarray = None):
        if around is not None:
            self.position = around + transforms.rotateY(self.position - around, angle)
        else:
            self.axis = transforms.rotateY(self.axis, angle)
    def rotateZ(self, angle: float, around: np.ndarray = None):
        if around is not None:
            self.position = around + transforms.rotateZ(self.position - around, angle)
        else:
            self.axis = transforms.rotateZ(self.axis, angle)
    def rotate(self, angle: float, axis: np.ndarray, around: np.ndarray = None):
        if around is not None:
            self.position = around + transforms.rotate(self.position - around, angle, axis)
        else:
            self.axis = transforms.rotate(self.axis, angle, axis)

    def getColor(self, point: np.ndarray):
        if self.material.texture is None:
            return self.material.color

        po = point - self.position
        n = self.getNormal(point)
        du = n @ self.__up
        dr = n @ self.__right
        angle = np.arccos(dr)
        if du < 0: angle = 2*np.pi - angle

        u = angle / (np.pi)
        v = np.linalg.norm(po)
        return self.material.texture.getColor(np.array([u, v]))
