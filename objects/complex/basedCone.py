import numpy as np
from objects import Cone, Circle
from utils.material import BLANK
from objects.complex import ComplexObject

class BasedCone(ComplexObject):
    def __init__(self, position: np.ndarray, axis: np.ndarray, height: float, radius: float, material=BLANK, vertice: np.ndarray = None):
        cone = Cone(position, axis, height, radius, material, vertice)
        circle = Circle(position - cone.axis * height, -axis, radius, material)
        super().__init__(
            position,
            [cone, circle],
            material
        )
