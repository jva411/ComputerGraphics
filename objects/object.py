import numpy as np
from utils.ray import Ray
from utils import transforms
from utils.material import Material


t_correction = 0.000001

class Object:
    def __init__(self, position: np.ndarray, material: Material):
        self.position = position
        # self.rotation = rotation/np.linalg.norm(rotation)
        # self.scale = scale
        self.isComplex = False
        self.isBVH = False
        self.bvhObject = None
        self.superObject = None
        self.material = material

    def preCalc(self):
        pass

    def intersects(self, ray: Ray) -> np.ndarray|None:
        return None

    def getNormal(self, point: np.ndarray) -> np.ndarray:
        return None

    def getColor(self, point: np.ndarray):
        return self.material.color

    def getDescription(self):
        return '%s\nX:%.2f Y:%.2f Z:%.2f' % (self.__class__.__name__, self.position[0], self.position[1], self.position[2])


    # Transform functions

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
