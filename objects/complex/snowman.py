import numpy as np
from utils import transforms
from objects import Sphere, Cone
from objects.complex.complexObjects import ComplexObject


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
