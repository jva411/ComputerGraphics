import numpy as np
from objects import Cylinder
from objects.sphere import Sphere
from utils.material import Lambertian, Texture
from objects.complex import ObjectComplex, BasedCone, BasedCylinder


TRUNK = Lambertian(color=[77., 37., 6.], shininess=100, texture=Texture('tree1.jpg', 0.03, False))
LEAF = Lambertian(color=[14., 100., 35.], shininess=10, texture=Texture('leaf1.jpg', 0.0025))
SPHERE = Lambertian(color=[255, 230, 80], shininess=2.)


class NatalTree(ObjectComplex):
    def __init__(self, position: np.ndarray):
        self.axis = np.array([0., 1., 0.])
        super().__init__(
            position,
            [
                BasedCylinder(np.array([0., 0., 0.]), -self.axis, 0.09, 0.30, material=TRUNK),
                Cylinder(np.array([0., 0.09, 0.]), -self.axis, 0.40, 0.06, material=TRUNK),
                BasedCone(np.array([0., 1.99, 0.]), self.axis, 1.5, 0.6, material=LEAF),
                Sphere(np.array([0., 2.0, 0.]), 0.045, material=SPHERE)
            ]
        )
        for part in self.parts:
            part.translate(position)
