import numpy as np
from utils.ray import Ray
from utils import transforms


t_correction = 0.00001

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
