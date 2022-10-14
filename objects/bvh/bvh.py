import numpy as np
from utils.ray import Ray
from objects import Object
from utils.material import BLANK
from objects.complex import ComplexObject


class BVH(ComplexObject):
    def __init__(self, bounding: Object, children: list[Object], material = BLANK):
        super().__init__(bounding.position, children, material)
        self.bounding = bounding
        self.isBVH = True

    def intersects(self, ray: Ray):
        return self.bounding.intersects(Ray(ray.origin, ray.direction, ray.t))
