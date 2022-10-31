import numpy as np
from utils.material import BLANK
from objects.object import Object


class ComplexObject(Object):
    def __init__(self, position: np.ndarray, parts: list[Object], material = BLANK):
        super().__init__(position, material)
        self.parts = parts
        self.isComplex = True

        for object in self.parts:
            object.superObject = self

    def translate(self, translation: np.ndarray):
        self.position += translation
        for part in self.parts:
            part.translate(translation)
