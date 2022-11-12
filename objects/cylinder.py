import math
import numba
import numpy as np
from utils.ray import Ray
from utils import transforms
from utils.material import BLANK
from objects.object import Object, t_correction


class Cylinder(Object):
    def __init__(self, position: np.ndarray, axis: np.ndarray, height: float, radius: float, material = BLANK, center_top: np.array = None):
        super().__init__(position, material)
        self.axis = transforms.normalize(axis if center_top is None else (position - center_top))
        self.height = height if center_top is None else np.linalg.norm(center_top - position)
        self.radius = radius

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
        return intersects(ray, self.position, self.axis, self.radius, self.height)

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


@numba.jit
def intersects(ray, position, axis, radius, height):
    v = ray.origin - position
    v = v - axis * (v @ axis)
    w = ray.direction - axis * (ray.direction @ axis)

    a = w @ w
    if (a==0): return None

    b = v @ w
    c = v @ v - radius ** 2
    delta = b ** 2 - a * c
    if delta < 0: return None

    delta2 = delta**0.5
    points = []
    t1 = (-b - delta2) / a - t_correction
    t2 = (-b + delta2) / a - t_correction
    p1 = ray.origin + ray.direction * t1
    p2 = ray.origin + ray.direction * t2
    dp1 = (position - p1) @ axis
    dp2 = (position - p2) @ axis

    if t1 > 0 and 0 <= dp1 <= height:
        points.append((t1, ray.origin + ray.direction * t1))
    if t2 > 0 and 0 <= dp2 <= height:
        points.append((t2, ray.origin + ray.direction * t2))
    if len(points) == 0: return None

    minPoint = points[0]
    for point in points:
        if point[0] < minPoint[0]:
            minPoint = point
    if ray.t < minPoint[0]: return None

    ray.t = minPoint[0]
    return minPoint[1]
