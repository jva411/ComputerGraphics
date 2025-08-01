import numpy as np
from objects.bvh import BVH
from objects.mesh import Cube
from utils.camera import Camera
from utils.material import Lambertian
from objects.complex import ObjectComplex, BasedCylinder


WOOD1 = Lambertian(color=[184., 108., 50.], shininess=100)
WOOD2 = Lambertian(color=[140., 78., 31.])


class Table(ObjectComplex):
    def __init__(self, position: np.ndarray, camera: Camera = None):
        top = Cube(material=WOOD1)
        top.scale(2.50, 0.05, 1.50)
        top.translate(np.array([0., 0.95, 0.]) + position)
        top.buildTriangles(camera)
        topBVH = BVH(BasedCylinder(
            np.array([top.center[0], top.position[1], top.center[2]]),
            None,
            None,
            top.radius,
            center_top=np.array([top.center[0], top.position[1] + top.scaled[1]*1.0001, top.center[2]])
        ), [top])

        legs = [Cube(material=WOOD2) for _ in range(2)]
        for leg in legs:
            leg.scale(0.05, 0.95, 1.50)
            leg.translate(position)

        legs[1].translate(2.45, 0., 0.)
        for leg in legs:
            leg.buildTriangles(camera)

        legsBVH = [
            BVH(BasedCylinder(
                np.array([leg.position[0], leg.center[1], leg.center[2]]),
                None,
                None,
                leg.radius,
                center_top=np.array([leg.position[0]+leg.scaled[0], leg.center[1], leg.center[2]]),
            ), [leg])
            for leg in legs
        ]

        super().__init__(
            position,
            [
                topBVH,
                *legsBVH
            ]
        )
