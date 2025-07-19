import numpy as np
from utils.ray import Ray
from utils import transforms
from utils.camera import Camera
from objects import Plane, Triangle
from objects.complex import ObjectComplex
from utils.material import BLANK, Material


class Mesh(ObjectComplex):
    def __init__(self, vertices: list[np.ndarray], edges: list[tuple[int, int]], faces: list[tuple[int, int, int]], material = BLANK):
        super().__init__(np.array([0., 0., 0.]), [], material)
        self.vertices = vertices
        self.edges = edges
        self.faces = faces
        self.isComplex = False
        self.isMesh = True
        self.scaled = np.array([1., 1., 1.])

    def buildTriangles(self, camera: Camera = None):
        self.center = sum(self.vertices) / len(self.vertices)
        self.radius = max((np.linalg.norm(v - self.center) for v in self.vertices))

        if camera is None:
            self.triangles = [self.createTriangle(face) for face in self.faces]
        else:
            normal = camera.direction if camera.perpendicular else transforms.normalize(self.center - camera.position)
            self.triangles = []
            for face in self.faces:
                t = self.createTriangle(face)
                d = t.normal @ normal
                if d < 0:
                    self.triangles.append(t)

        self.isComplex = True
        self.parts = self.triangles

    def translate(self, x: np.ndarray|float, y: float = None, z: float = None):
        vector = x if y is None else np.array([x, y, z])
        for vertice in self.vertices:
            vertice += vector

        self.position += vector
        return self

    def scale(self, x: np.ndarray|float, y: float = None, z: float = None, point: np.ndarray = None):
        vector = x if y is None else np.array([x, y, z])

        self.scaled *= vector
        point = point or self.position
        for vertice in self.vertices:
            vertice -= point
            vertice *= vector
            vertice += point

        return self

    def __reflectAxis(self, axis: int):
        for vertice in self.vertices:
            vertice[axis] = -vertice[axis]
        for i in range(len(self.faces)):
            self.faces[i] = self.faces[i][::-1]

        self.position[axis] = -self.position[axis]
        return self

    def reflectX(self):
        return self.__reflectAxis(2)
    def reflectY(self):
        return self.__reflectAxis(1)
    def reflectZ(self):
        return self.__reflectAxis(0)

    def reflect(self, plane: Plane):
        for vertice in self.vertices:
            ray = Ray(vertice, -plane.normal)
            plane.intersects(ray)
            vertice += ray.direction * (2*ray.t)
        for i in range(len(self.faces)):
            self.faces[i] = self.faces[i][::-1]

        ray = Ray(self.position, -plane.normal)
        plane.intersects(ray)
        self.position += ray.direction * (2*ray.t)
        return self

    def __shearAB(self, angle, a, b, isTan):
        for vertice in self.vertices:
            vertice -= self.position
            vertice[a] += vertice[b] * (angle if isTan else np.tan(angle))
            vertice += self.position

        self.position[a] += np.tan(angle) * self.position[b]
        return self

    def shearXY(self, angle, isTan=False):
        return self.__shearAB(angle, 0, 1, isTan)
    def shearXZ(self, angle, isTan=False):
        return self.__shearAB(angle, 0, 2, isTan)
    def shearYX(self, angle, isTan=False):
        return self.__shearAB(angle, 1, 0, isTan)
    def shearYZ(self, angle, isTan=False):
        return self.__shearAB(angle, 1, 2, isTan)
    def shearZX(self, angle, isTan=False):
        return self.__shearAB(angle, 2, 0, isTan)
    def shearZY(self, angle, isTan=False):
        return self.__shearAB(angle, 2, 1, isTan)

    def __rotate(self, rotation, *args):
        for vertice in self.vertices:
            vertice -= self.position
            rotation(vertice, *args, changeVector=True)
            vertice += self.position
        return self

    def rotateX(self, angle):
        return self.__rotate(transforms.rotateX, angle)
    def rotateY(self, angle):
        return self.__rotate(transforms.rotateY, angle)
    def rotateZ(self, angle):
        return self.__rotate(transforms.rotateZ, angle)
    def rotate(self, angle, axis):
        return self.__rotate(transforms.rotate, angle, axis)

    def createTriangle(self, face: tuple[int, int, int]):
        AB = self.edges[face[0]]
        BC = self.edges[face[1]]
        if not AB[0] == BC[0] and not AB[0] == BC[1]:
            A = self.vertices[AB[0]]
            B = self.vertices[AB[1]]
            C = self.vertices[BC[1]] if BC[0] == AB[1] else self.vertices[BC[0]]
        else:
            A = self.vertices[AB[1]]
            B = self.vertices[AB[0]]
            C = self.vertices[BC[1]] if BC[0] == AB[0] else self.vertices[BC[0]]

        t = Triangle(A, B, C, self.material)
        t.superObject = self
        return t

    def copy(self):
        m = Mesh(
            [v.copy() for v in self.vertices],
            [(e[0], e[1]) for e in self.edges],
            [(f[0], f[1], f[2]) for f in self.faces],
            self.material.copy()
        )
        m.position = self.position.copy()
        return m
