import math
import numpy as np
from utils.ray import Ray
from utils import transforms


class Object:
    def __init__(self, position: np.ndarray, color: np.ndarray):
        self.position = position
        # self.rotation = rotation/np.linalg.norm(rotation)
        # self.scale = scale
        self.color = color
        self.isComplex = False
        self.superObject = None

    def intersects(self, ray: Ray) -> np.ndarray|None:
        return None

    def getNormal(self, point: np.ndarray) -> np.ndarray:
        return None

    def translate(self, translation: np.ndarray):
        self.position += translation

    def rotateX(self, angle: float, around: np.ndarray = None):
        if around is not None:
            self.position = around + transforms.rotateX(self.position - around, angle)
    def rotateY(self, angle: float, around: np.ndarray = None):
        if around is not None:
            self.position = around + transforms.rotateY(self.position - around, angle)
    def rotateZ(self, angle: float, around: np.ndarray = None):
        if around is not None:
            self.position = around + transforms.rotateZ(self.position - around, angle)
    def rotate(self, angle: float, axis: np.ndarray, around: np.ndarray = None):
        if around is not None:
            self.position = around + transforms.rotate(self.position - around, angle, axis)


class Sphere(Object):
    def __init__(self, position: np.ndarray, radius: float, color: np.ndarray = np.array([255., 255., 255.])):
        super().__init__(position, color)
        self.radius = radius


    def intersects(self, ray: Ray) -> np.ndarray:
        m: np.ndarray = ray.origin - self.position
        b = m @ ray.direction
        c = m @ m - self.radius ** 2
        if c > 0 and b > 0: return None

        delta = b ** 2 - c
        if delta < 0: return None

        distance = -b - np.sqrt(delta)
        if distance < 0 or ray.t < distance: return None

        ray.t = distance
        return ray.hitting_point

    def getNormal(self, point: np.ndarray) -> np.ndarray:
        return transforms.normalize(point - self.position)


class Cone(Object):
    def __init__(self, position: np.ndarray, axis: np.ndarray, height: float, radius: float, color = np.array([255., 255., 255.])):
        super().__init__(position, color)
        self.axis = transforms.normalize(axis)
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
        t1 = (-b - sqrtDelta) / a
        t2 = (-b + sqrtDelta) / a
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
            print(self.position, axis, around, math.degrees(angle))
            self.position = around + transforms.rotate(self.position - around, angle, axis)
            print(self.position)
        else:
            self.axis = transforms.rotate(self.axis, angle, axis)


class Plane(Object):
    def __init__(self, position: np.ndarray, normal: np.ndarray, color: np.ndarray = np.array([255., 255., 255.])):
        super().__init__(position, color)
        self.normal = transforms.normalize(normal)

    def intersects(self, ray: Ray) -> np.ndarray:
        dn = ray.direction @ self.normal
        if dn == 0: return None

        t = (self.position - ray.origin) @ self.normal / dn
        if t < 0 or ray.t < t: return None

        ray.t = t
        return ray.hitting_point

    def getNormal(self, point: np.ndarray) -> np.ndarray:
        return self.normal


class Circle(Plane):
    def __init__(self, position: np.ndarray, normal: np.ndarray, radius: float, color: np.ndarray = np.array([255., 255., 255.])):
        super().__init__(position, normal, color)
        self.radius = radius
        print(self.position, self.normal, self.radius)

    def intersects(self, ray: Ray) -> np.ndarray:
        dn = ray.direction @ self.normal
        if dn == 0: return None

        t = (self.position - ray.origin) @ self.normal / dn
        if t < 0 or ray.t < t: return None

        distance = np.linalg.norm((ray.origin + ray.direction * t) - self.position)
        if distance > self.radius: return None

        ray.t = t
        return ray.hitting_point


class ComplexObject(Object):
    def __init__(self, position: np.ndarray, parts: list[Object]):
        super().__init__(position, None)
        self.parts = parts


class Snowman(ComplexObject):
    def __init__(self, position: np.ndarray):
        self.axis = np.array([0., 1., 0.])
        self.heights = [0., 1.2, 2.25, 2.3]
        super().__init__(
            position,
            [
                Sphere(np.array([0., 0.8, 0.]), .8),
                Sphere(np.array([0., 2.0, 0.]), .8),
                Sphere(np.array([0., 3.05, 0.]), .5),
                Cone(np.array([0., 3.1, -1.]), np.array([0, 0, -1]), 0.6, .1, np.array([226, 146, 100])),
            ]
        )
        self.isComplex = True
        for part in self.parts:
            part.superObject = self
            part.translate(position)

    def rotate(self, angle: float, axis: np.ndarray, around: np.ndarray = None):
        if around is not None:
            self.position = around + transforms.rotate(self.position - around, angle, axis)
            for part in self.parts:
                part.rotate(angle, axis, around)
        else:
            self.axis = transforms.rotate(self.axis, angle, axis)
            for index, part in enumerate(self.parts):
                part.rotate(angle, axis, None)
                part.rotate(angle, axis, self.position + self.axis * self.heights[index])


class BasedCone(ComplexObject):
    def __init__(self, position: np.ndarray, axis: np.ndarray, height: float, radius: float, color: np.ndarray = np.array([255., 255., 255.])):
        self.axis = transforms.normalize(axis)
        self.radius = radius
        self.height = height
        super().__init__(
            position,
            [
                Cone(position, self.axis, height, radius, color),
                Circle(position - self.axis * height, -self.axis, radius, color),
            ]
        )
        self.isComplex = True
        for part in self.parts:
            part.superObject = self
