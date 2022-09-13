import math
import numpy as np
from utils.ray import Ray
from utils import transforms
from utils.material import BLANK
from objects.object import Object, t_correction


class Cone(Object):
    def __init__(self, position: np.ndarray, axis: np.ndarray, height: float, radius: float, material=BLANK, vertice: np.ndarray = None):
        super().__init__(position, material)
        self.axis = transforms.normalize(axis if not vertice else (vertice - position))
        self.height = height
        self.radius = radius
        self.__hip = math.sqrt(height ** 2 + radius ** 2)
        self.__cos = height / self.__hip
        self.__cos2 = self.__cos ** 2

    def intersects(self, ray: Ray) -> np.ndarray:
        v = self.position - ray.origin

        dn = ray.direction @ self.axis
        vn = v @ self.axis

        a = (dn**2) - (ray.direction @ ray.direction * self.__cos2)
        b = (v @ ray.direction * self.__cos2) - (vn * dn)
        c = (vn**2) - (v @ v * self.__cos2)
        delta = b**2 - a*c

        points = []

        if delta < 0: return None  # TODO: check intersecation with the base

        sqrtDelta = math.sqrt(delta)
        t1 = (-b - sqrtDelta) / a - t_correction
        t2 = (-b + sqrtDelta) / a - t_correction
        p1 = ray.origin + ray.direction * t1
        p2 = ray.origin + ray.direction * t2
        dp1 = (self.position - p1) @ self.axis
        dp2 = (self.position - p2) @ self.axis

        if t1 > 0 and 0 <= dp1 <= self.height:
            points.append((t1, p1))

        if t2 > 0 and 0 <= dp2 <= self.height:
            points.append((t2, p2))

        if len(points) == 0: return None

        minPoint = min(points, key=lambda x: x[0])
        if ray.t < minPoint[0]: return None

        ray.t = minPoint[0]
        return minPoint[1]

    def getNormal(self, point: np.ndarray) -> np.ndarray:
        v = transforms.normalize(self.position - point)
        n = self.axis - v*self.__cos
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
