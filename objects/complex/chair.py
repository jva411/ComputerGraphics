import numpy as np
from objects.bvh import BVH
from objects import Cylinder
from objects.mesh import Cube
from utils.camera import Camera
from utils.material import Material
from objects.complex import ObjectComplex, BasedCylinder


# WOOD1 = Material(color=[184., 108., 50.], shininess=100)
# WOOD2 = Material(color=[140., 78., 31.])


class Chair(ObjectComplex):
    def __init__(self, position: np.ndarray, camera: Camera = None):
        axis = np.array([0., -1., 0.])
        legHeight = 0.4
        legRadius = 0.02
        legPosCorrection = legRadius * 1.42
        seatSize = 0.5
        legs = [Cylinder(position.copy(), axis, legHeight, legRadius) for _ in range(4)]
        legs[0].translate(np.array([ legPosCorrection, 0.,  legPosCorrection]))
        legs[1].translate(np.array([ legPosCorrection, 0., -legPosCorrection + seatSize]))
        legs[2].translate(np.array([-legPosCorrection + seatSize, 0.,  legPosCorrection]))
        legs[3].translate(np.array([-legPosCorrection + seatSize, 0., -legPosCorrection + seatSize]))

        seat = Cube()
        seat.scale(seatSize, 0.05, seatSize)
        seat.translate(position)
        seat.translate(0., legHeight, 0.)
        seat.buildTriangles(camera)
        seatBVH = BVH(BasedCylinder(
            seat.position + np.array([seatSize/2, -0.001, seatSize/2]),
            axis,
            0.051,
            seatSize * 1.42 / 2
        ), [seat])

        backseat = seat.copy()
        backseat.rotateZ(np.radians(90))
        backseat.translate(0., 0.20, 0.)
        backseat.buildTriangles(camera)
        backseatBVH = BVH(BasedCylinder(
            backseat.position + np.array([0.001, seatSize/2, seatSize/2]),
            np.array([1., 0., 0.]),
            0.051,
            seatSize * 1.42 / 2
        ), [backseat])

        back_supports = [Cylinder(backseat.position + np.array([-0.010, -0.20, 0.10]), axis, 0.20, 0.015) for _ in range(2)]
        back_supports[1].translate(np.array([0., 0., 0.25]))

        right_support1 = BasedCylinder(backseat.position + np.array([-0.025, 0.06, -0.10]), np.array([0., 0., -1.]), 0.10, 0.01, baseTop=False)
        right_support2 = BasedCylinder(backseat.position + np.array([-0.025, 0.06, -0.10]), np.array([-1., 0., 0.]), 0.25, 0.01, baseTop=False)

        table1 = Cube()
        table1.scale(0.20, 0.02, 0.10)
        table1.translate(right_support1.position + np.array([0.20, -0.01, -0.05]))
        table1.buildTriangles(camera)
        table1BVH = BVH(BasedCylinder(table1.position + np.array([0.10, 0.0201, 0.05]), -axis, 0.021, (0.1**2 + 0.05**2 + 0.001) ** 0.5), [table1])

        table2 = table1.copy()
        table2.scale(2., 1., 3.)
        table2.translate(0.20, 0., 0.)
        table2.buildTriangles(camera)
        table2BVH = BVH(BasedCylinder(table2.position + np.array([0.20, 0.0201, 0.15]), -axis, 0.021, (0.2**2 + 0.15**2 + 0.001) ** 0.5), [table2])

        super().__init__(position, [
            *legs,
            *back_supports,
            seatBVH,
            backseatBVH,
            right_support1,
            right_support2,
            table1BVH,
            table2BVH
        ])

        self.center = self.position + np.array([0.30, 0.575, 0.25])
        self.radius = 0.68
