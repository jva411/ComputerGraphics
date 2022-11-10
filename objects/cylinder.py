import math
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
        v = ray.origin - self.position
        v = v - self.axis * (v @ self.axis)
        w = ray.direction - self.axis * (ray.direction @ self.axis)

        a = w @ w
        b = v @ w
        c = v @ v - self.radius ** 2
        delta = b ** 2 - a * c
        if delta < 0: return None

        points = []
        t1 = (-b - math.sqrt(delta)) / a - t_correction
        t2 = (-b + math.sqrt(delta)) / a - t_correction
        p1 = ray.origin + ray.direction * t1
        p2 = ray.origin + ray.direction * t2
        dp1 = (self.position - p1) @ self.axis
        dp2 = (self.position - p2) @ self.axis

        if t1 > 0 and 0 <= dp1 <= self.height:
            points.append((t1, ray.origin + ray.direction * t1))
        if t2 > 0 and 0 <= dp2 <= self.height:
            points.append((t2, ray.origin + ray.direction * t2))
        if len(points) == 0: return None

        minPoint = min(points, key=lambda x: x[0])
        if ray.t < minPoint[0]: return None

        ray.t = minPoint[0]
        return minPoint[1]

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
