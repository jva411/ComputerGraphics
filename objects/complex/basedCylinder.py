import numpy as np
from objects import Cylinder, Circle
from utils.material import BLANK
from objects.complex import ComplexObject

class BasedCylinder(ComplexObject):
    def __init__(self, position: np.ndarray, axis: np.ndarray, height: float, radius: float, material = BLANK, center_top: np.array = None, baseTop=True):
        cylinder = Cylinder(position, axis, height, radius, material, center_top)
        base = Circle(position, -axis, radius, material)
        parts = [cylinder, base]
        if baseTop:
            parts.append(Circle(position - cylinder.axis * height, -axis, radius, material))

        super().__init__(position, parts, material)
