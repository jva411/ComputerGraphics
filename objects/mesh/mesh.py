import numpy as np
from utils.ray import Ray
from utils.material import BLANK
from objects.complex import ComplexObject
from objects import Plane, Triangle as ObjectTriangle


class Mesh(ComplexObject):
    def __init__(self, vertices: list[np.ndarray], edges: list[tuple[int, int]], faces: list[tuple[int, int, int]], material = BLANK):
        super().__init__(np.array([0., 0., 0.]), [], material)
        self.vertices = vertices
        self.edges = edges
        self.faces = faces
        self.isComplex = False
        self.scaled = np.array([1., 1., 1.])

    def buildTriangles(self, normal: np.ndarray = None):
        if normal is None:
            self.triangles = [Triangle(self, face) for face in self.faces]
        else:
            self.triangles = []
            for face in self.faces:
                t = Triangle(self, face)
                d = t.normal @ normal
                if d < 0:
                    self.triangles.append(t)

        self.isComplex = True
        self.parts = self.triangles
        self.center = sum(self.vertices) / len(self.vertices)
        self.radius = max((np.linalg.norm(v - self.center) for v in self.vertices))

    def translate(self, x: np.ndarray|float, y: float = None, z: float = None):
        vector = x if y is None else np.array([x, y, z])

        # print('Translate', vector)
        for vertice in self.vertices:
            vertice += vector
            # print(vertice)

        self.position += vector
        # print('=' + '-='*20 + '\n')
        return self

    def scale(self, x: np.ndarray|float, y: float = None, z: float = None, point: np.ndarray = None):
        vector = x if y is None else np.array([x, y, z])

        # print('Scale', vector, self.position)
        self.scaled *= vector
        point = point or self.position
        for vertice in self.vertices:
            # print(vertice)
            vertice -= point
            vertice *= vector
            vertice += point
            # print(vertice)

        # print('=' + '-='*20 + '\n')
        return self

    def __reflectAxis(self, axis: int):
        for vertice in self.vertices:
            vertice[axis] = -vertice[axis]

        self.position[axis] = -self.position[axis]
        return self

    def reflectX(self):
        return self.__reflectAxis(0)
    def reflectY(self):
        return self.__reflectAxis(1)
    def reflectZ(self):
        return self.__reflectAxis(2)

    def reflect(self, plane: Plane):
        for vertice in self.vertices:
            ray = Ray(vertice, -plane.normal)
            plane.intersects(ray)
            vertice += ray.direction * (2*ray.t)

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


class Triangle(ObjectTriangle):
    def __init__(self, mesh: Mesh, face: tuple[int, int, int]):
        AB = mesh.edges[face[0]]
        BC = mesh.edges[face[1]]
        if not AB[0] == BC[0] and not AB[0] == BC[1]:
            self.A = mesh.vertices[AB[0]]
            self.B = mesh.vertices[AB[1]]
            self.C = mesh.vertices[BC[1]] if BC[0] == AB[1] else mesh.vertices[BC[0]]
        else:
            self.A = mesh.vertices[AB[1]]
            self.B = mesh.vertices[AB[0]]
            self.C = mesh.vertices[BC[1]] if BC[0] == AB[0] else mesh.vertices[BC[0]]

        super().__init__(self.A, self.B, self.C, mesh.material)
        self.superObject = mesh
