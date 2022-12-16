import math
import pygame
import numpy as np
from objects.bvh import BVH
from utils import transforms
from threading import Thread
from utils.scene import Scene
from utils.window import Window
from utils.camera import Camera
from console.console import Console
from objects.mesh import Cube, Ramp
from objects.complex.table import Table
from objects.complex.chair import Chair
from utils.material import Material, Texture
from objects.complex.shed import Portico, Shed
from objects import Sphere, Cone, Plane, Cylinder, Triangle
from objects.complex import Snowman, Tree, BasedCone, BasedCylinder, NatalTree
from lights.lights import AmbientLight, PointLight, DirectionalLight, SpotLight


def main():
    w, h = 400, 300
    camera_pos = np.array([0., 3.5, -6.0])
    camera_at = np.array([0., 1., 0.])
    camera = Camera(
        (w, h),
        camera_pos,
        camera_at,
        n_threads=2,
        windowSize=np.array([600., 450.])
    )

    stairHeight = 0.30
    stairsMaterial = Material(np.array([200., 200., 200.]), 1.0)
    stairs = [Cube(stairsMaterial) for _ in range(4)]
    for i in range(len(stairs)):
        stairs[i].scale(4., stairHeight, 1.0)
        stairs[i].translate(-2.0, stairHeight * i, -1.0 * (i+1))
        stairs[i].buildTriangles(camera)
    stairsBVH = [BVH(BasedCylinder(
        s.position + np.array([-0.001, 0.125, 0.5]),
        np.array([-1., 0., 0.]),
        4.1,
        ((0.125 ** 2) + (0.5 ** 2) + 0.1) ** 0.5
    ), [s]) for s in stairs]

    rampsMaterial = Material(np.array([237., 228., 198.]), 1.0)
    n = len(stairs)
    ramp1 = Ramp(rampsMaterial)
    ramp1.rotateY(np.radians(270))\
         .scale(1.0, stairHeight*n, 1.0*n)\
         .translate(-2.0, 0., -1.0*n)\
         .buildTriangles(camera)
    ramp1BVH = BVH(BasedCylinder(
        ramp1.position + np.array([-0.5, stairHeight * n / 2, -0.001]),
        None, None, ((stairHeight*n) ** 2 + 0.5 ** 2 + 0.001) ** 0.5,
        center_top=ramp1.position + np.array([-0.5, 0., n+0.001])
    ), [ramp1])
    ramp2 = ramp1.copy()
    ramp2.reflectZ().buildTriangles(camera)
    ramp2BVH = BVH(BasedCylinder(
        ramp2.position + np.array([0.5, stairHeight * n / 2, -0.001]),
        None, None, ((stairHeight*n) ** 2 + 0.5 ** 2 + 0.001) ** 0.5,
        center_top=ramp2.position + np.array([0.5, 0., n+0.001])
    ), [ramp2])

    chairs = []
    for i in range(3):
        chairs = [*chairs, *[Chair(np.array([-1.0 -0.8*j, 0.30*(i+1), -1.0*(i+1)]), camera) for j in range(2)]]
        chairs = [*chairs, *[Chair(np.array([1.3 -0.8*j, 0.30*(i+1), -1.0*(i+1)]), camera) for j in range(2)]]
    chairsBVH = [BVH(Sphere(chair.center, chair.radius), [chair]) for chair in chairs]

    wallMaterial = Material(np.array([103., 136., 142.]), 100)
    walls = [Cube(wallMaterial) for _ in range(3)]
    for wall in walls:
        wall.scale(0.10, 5.0, 9.0)
        wall.translate(3.0, 0., -6.0)
    walls[1].translate(-6.10, 0., 0.)
    walls[2].rotateY(np.radians(90)).scale(6./9., 1., 1.).translate(-6., 0., 9.)

    floorMaterial = Material(np.array([155., 155., 155.]))
    floor1 = Cube(floorMaterial)\
        .scale(6.0, 0.10, 3.0)\
        .translate(-3.0, 0., 0.0)
    floor1.buildTriangles(camera)
    floor2 = Cube(floorMaterial)\
        .scale(6.0, 0.10, 2.0)\
        .translate(-3.0, stairHeight * n, -1.0 * n - 2.0)
    floor2.buildTriangles(camera)
    floor3 = Cube(floorMaterial)\
        .scale(6.0, 0.10, 9.0)\
        .translate(-3.0, 5.0, -6.0)
    floor3.buildTriangles(camera)
    floors = [floor1, floor2, floor3]

    boardMaterial = Material(np.array([11., 45., 21.]), 0.1)
    board = Cube(boardMaterial)\
        .scale(5.0, 1.0, 0.05)\
        .translate(-2.5, 1.40, 2.90)
    board.buildTriangles(camera)

    stage = floor1.copy().scale(1., 6., 2./3.).translate(0., 0., 1.0)
    stage.material = stairsMaterial
    stage.buildTriangles(camera)
    stageBVH = BVH(BasedCylinder(stage.position + np.array([0., 0.30, 1.0]), np.array([-1., 0., 0.]), 6.0, (0.30 ** 2 + 1.0 ** 2 + 0.001) ** 0.5), [stage])

    snowman = Snowman(np.array([-2.3, 0.60, 2.0]), 1/2.5)
    snowman.rotateY(math.radians(20))
    table = Table(np.array([-1.7, 0.60, 1.1]), camera)
    natalTree = NatalTree(np.array([2.5, 0.60, 1.5]))
    plane = Plane(np.array([0., 0., 0.]), np.array([0., 1., 0.]), Material(None, texture=Texture('grass.jpeg', 0.0002, False)))

    objects = [
        snowman,
        table,
        natalTree,
        plane,
        # *chairsBVH,
        # *stairsBVH,
        # ramp1BVH,
        # ramp2BVH,
        stageBVH,
        board,
        *walls,
        # *floors,
    ]
    spotPos = np.array([0., 3.0, 0.])
    lights = [
        SpotLight(spotPos, -(snowman.position - spotPos), 0.8, np.radians(20), color=np.array([230., 230., 150.])),
        PointLight(np.array([-2.5, 3.0, 2.0]), 0.7),
        PointLight(np.array([2.5, 3.0, 2.0]), 0.7),
        DirectionalLight(np.array([-1., -1., 1.0]), 0.35),
        AmbientLight(0.10)
    ]
    scene = Scene(w, h, camera, objects, lights)
    window = Window(scene, title="Cube")

    def consoleInit():
        Console(window, 'main>').cmdloop()

    Thread(target=consoleInit, daemon=True).start()
    window.open()
    window.startLoop()


if __name__ == "__main__":
    main()
