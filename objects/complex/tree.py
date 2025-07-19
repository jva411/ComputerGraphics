import numpy as np
from objects import Cylinder, Cone
from objects.complex import ObjectComplex
from utils.material import Lambertian, Texture


TRUNK = Lambertian(color=[77., 37., 6.], shininess=100, texture=Texture('tree1.jpg', 0.003, False))
LEAF = Lambertian(color=[14., 100., 35.], shininess=10, texture=Texture('leaf1.jpg', 0.0005))


class Tree(ObjectComplex):
    def __init__(self, position: np.ndarray):
        self.axis = np.array([0., 1., 0.])
        super().__init__(
            position,
            [
                Cylinder(np.array([0., 0., 0.]), -self.axis, 5., 0.7, material=TRUNK),
                Cone(np.array([0., 5., 0.]), self.axis, 1.5, 1.5, LEAF),
                Cone(np.array([0., 6., 0.]), self.axis, 1.5, 1.5, LEAF)
            ]
        )
        self.isComplex = True
        for part in self.parts:
            part.superObject = self
            part.translate(position)
