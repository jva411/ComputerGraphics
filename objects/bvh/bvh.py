import numpy as np
from utils.ray import Ray
from objects import Object
from utils.material import BLANK
from objects.complex.objectComplex import ObjectComplex


class BVH(ObjectComplex):
    def __init__(self, bounding: Object, children: list[Object], material = BLANK):
        assert (bounding.isBVH is False), 'BVH cannot have another BVH object as bounding'
        super().__init__(bounding.position, children, material)
        self.bounding = bounding
        self.bounding.bvhObject = self
        self.isBVH = True

    def intersects(self, ray: Ray):
        return self.bounding.intersects(Ray(ray.origin, ray.direction, ray.t))

    def translate(self, translation: np.ndarray):
        self.position += translation
        self.bounding.translate(translation)
        for part in self.parts:
            part.translate(translation)

    def preCalc(self, reverse=False):
        super().preCalc(reverse)
        self.bounding.preCalc(reverse)
