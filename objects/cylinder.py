import math
import numpy as np
from utils.ray import Ray
from utils import transforms
from objects.object import Object


class Cylinder(Object):
    def __init__(self, position: np.ndarray, axis: np.ndarray, height: float, radius: float, color: np.ndarray = np.array([255., 255., 255.])):
        super().__init__(position, color)
        self.axis = transforms.normalize(axis)
        self.height = height
        self.radius = radius

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
        t1 = (-b - math.sqrt(delta)) / a
        t2 = (-b + math.sqrt(delta)) / a
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
        n = self.axis - w * (w @ self.axis)
        return transforms.normalize(n)
