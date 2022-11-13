import numpy as np
from objects import Cylinder, Circle
from utils.material import BLANK
from objects.complex import ObjectComplex


class BasedCylinder(ObjectComplex):
    def __init__(self, position: np.ndarray, axis: np.ndarray, height: float, radius: float, material = BLANK, center_top: np.array = None, baseTop=True):
        axis = axis if center_top is None else (position - center_top)

        cylinder = Cylinder(position.copy(), axis.copy(), height, radius, material, center_top)
        base = Circle(position.copy(), -cylinder.axis.copy(), radius, material)
        parts = [cylinder, base]
        if baseTop:
            parts.append(Circle(position - cylinder.axis * cylinder.height, -cylinder.axis.copy(), radius, material))

        super().__init__(position, parts, material)
