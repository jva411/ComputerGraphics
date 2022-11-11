import math
import numba
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

    def intersects(self, ray: Ray) -> np.ndarray:
        t, point = intersects(ray.origin, ray.direction, ray.t, self.position, self.axis, self.height, self.__cos2)
        if t is not None:
            ray.t = t

        return point

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


@numba.jit(nopython=True)
def intersects(rayOrigin, rayDirection, maxT, position, axis, height, cos2):
    v = position - rayOrigin

    dn = rayDirection @ axis
    vn = v @ axis

    a = (dn**2) - (rayDirection @ rayDirection * cos2)
    if a==0: return None, None

    b = (v @ rayDirection * cos2) - (vn * dn)
    c = (vn**2) - (v @ v * cos2)
    delta = b**2 - a*c

    if delta < 0: return None, None

    points = []
    sqrtDelta = math.sqrt(delta)
    t1 = (-b - sqrtDelta) / a - t_correction
    t2 = (-b + sqrtDelta) / a - t_correction
    p1 = rayOrigin + rayDirection * t1
    p2 = rayOrigin + rayDirection * t2
    dp1 = (position - p1) @ axis
    dp2 = (position - p2) @ axis

    if t1 > 0 and 0 <= dp1 <= height:
        points.append((t1, p1))

    if t2 > 0 and 0 <= dp2 <= height:
        points.append((t2, p2))

    if len(points) == 0: return None, None

    minPoint = points[0]
    for point in points:
        if point[0] < minPoint[0]:
            minPoint = point
    if maxT < minPoint[0]: return None, None

    t, point = minPoint
    return t, point
