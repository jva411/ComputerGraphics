import numpy as np
from itertools import chain
from objects.bvh import BVH
from objects.mesh import Cube
from utils.material import Material, Texture
from objects.complex import ComplexObject, BasedCylinder


WOOD = Material(color=[170., 115., 24.], shininess=100)
ROOFTOP = Material(color=[255., 90., 0.], shininess=10)


class Portico(ComplexObject):
    def __init__(self, position: np.ndarray, cameraDirection: np.ndarray = None):
        girders = [Cube(material=WOOD) for _ in range(2)]
        for girder in girders:
            girder.scale(0.50, 5.00, 0.30)

        girders[1].translate(6.50, 0., 0.)
        bvhGirders = [BVH(
            BasedCylinder(
                np.array([0.25, 0., 0.15]) + girder.position,
                None,
                None,
                0.35,
                center_top=np.array([0.25, 5., 0.15]) + girder.position
            ), [girder]
        ) for girder in girders]

        shearedGirders = [Cube(material=WOOD) for _ in range(2)]
        for girder in shearedGirders:
            girder.scale(3.00, 0.50, 0.30)
            girder.translate(0.50, 4.50, 0.)

        shearedGirders[0].shearYX(0.75, True)
        shearedGirders[1].translate(3.00, 2.25, 0.)
        shearedGirders[1].shearYX(-0.75, True)
        axis1, axis2 = np.array([3., 2.25, 0.]), np.array([3., -2.25, 0.])
        bvhShearedGirders = [BVH(
            BasedCylinder(np.array([0.50, 4.75, 0.15]) -axis1*0.05, -axis1, 4.2, 0.35),
            [shearedGirders[0]]
        ), BVH(
            BasedCylinder(np.array([3.50, 7.00, 0.15]) -axis2*0.05, -axis2, 4.2, 0.35),
            [shearedGirders[1]]
        )]

        for part in chain(girders, shearedGirders):
            part.buildTriangles(cameraDirection)
            # for v in part.vertices:
            #     print(v)

            # print('=' * 30 + '\n\n')

        super().__init__(
            position,
            [
                *bvhGirders,
                *bvhShearedGirders,
                # *[b.bounding for b in bvhShearedGirders2]
            ]
        )
        for part in self.parts:
            part.translate(position)


class Shed(ComplexObject):
    def __init__(self, position: np.ndarray, cameraDirection: np.ndarray = None):
        porticos = [Portico(np.array([0., 0., 0.]), cameraDirection) for _ in range(2)]
        porticos[1].translate(np.array([0., 0., 10.50]))

        rooftops = [Cube(material=ROOFTOP) for _ in range(2)]
        for rooftop in rooftops:
            rooftop.scale(3.50, 0.05, 10.00)
            rooftop.translate(0., 4.60, 0.30)

        rooftops[0].shearYX(0.75, True)
        rooftops[1].translate(3.50, 2.625, 0.)
        rooftops[1].shearYX(-0.75, True)
        axis1, axis2 = np.array([-2.25, 3., 0.]), np.array([-2.25, -3., 0.])
        bvhRooftops = [BVH(
            BasedCylinder(np.array([1.75, 5.75, 5.30]) -axis1*0.05, -axis1, 0.7, 7.1),
            [rooftops[0]]
        ), BVH(
            BasedCylinder(np.array([5.25, 5.75, 5.30]) -axis2*0.05, -axis2, 0.7, 7.1),
            [rooftops[1]]
        )]

        for part in rooftops:
            part.buildTriangles(cameraDirection)

        lrWalls = [Cube(material=WOOD) for _ in range(2)]
        for lrWall in lrWalls:
            lrWall.scale(0.20, 4.50, 10.00)
            lrWall.translate(0.30, 0., 0.30)

        lrWalls[1].translate(6.20, 0., 0.)
        axis = np.array([1., 0., 0.])
        bvhLrWalls = [BVH(
            BasedCylinder(np.array([0., 2.25, 5.0]) + lr.position -axis*0.05, -axis, 0.3, 7.1),
            [lr]
        ) for lr in lrWalls]

        for part in lrWalls:
            part.buildTriangles(cameraDirection)

        backWall = Cube(material=WOOD)
        backWall.scale(6.00, 4.50, 0.20)
        backWall.translate(0.50, 0., 10.30)
        axis = np.array([0., 0., 1.])
        bvhBackWall = BVH(BasedCylinder(
            np.array([3.50, 2.25, 10.30]) -axis*0.05,
            axis,
            0.3,
            8.6
        ), [backWall])
        backWall.buildTriangles(cameraDirection)

        super().__init__(
            position,
            [
                *porticos,
                *bvhRooftops,
                *bvhLrWalls,
                bvhBackWall
            ]
        )
        for part in self.parts:
            part.translate(position)
