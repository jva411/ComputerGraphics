import numpy as np
from objects import Cone, Circle
from utils.material import BLANK
from objects.complex import ObjectComplex

class BasedCone(ObjectComplex):
    def __init__(self, position: np.ndarray, axis: np.ndarray, height: float, radius: float, material=BLANK, vertice: np.ndarray = None):
        cone = Cone(position.copy(), axis.copy(), height, radius, material, vertice and vertice.copy())
        circle = Circle(position - cone.axis * height, -axis.copy(), radius, material)

        super().__init__(position, [cone, circle], material)
