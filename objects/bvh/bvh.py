import numpy as np
from utils.ray import Ray
from objects import Object
from utils.material import BLANK
from objects.complex.complexObjects import ComplexObject


class BVH(ComplexObject):
    def __init__(self, bounding: Object, children: list[Object], material = BLANK):
        assert (bounding.isBVH is False), 'BVH cannot have another BVH object as bounding'
        super().__init__(bounding.position, children, material)
        self.bounding = bounding
        self.isBVH = True

    def intersects(self, ray: Ray):
        return self.bounding.intersects(Ray(ray.origin, ray.direction, ray.t))
