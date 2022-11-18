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
from utils.material import Material, Texture
from objects.complex.shed import Portico, Shed
from objects import Sphere, Cone, Plane, Cylinder, Triangle
from lights.lights import AmbientLight, PointLight, DirectionalLight
from objects.complex import Snowman, Tree, BasedCone, BasedCylinder, NatalTree


def main():
    w, h = 800, 600
    camera_pos = np.array([-1., 2.0, -6.0])
    camera_at = np.array([-1., 1., 0.])
    camera = Camera(
        (w, h),
        camera_pos,
        camera_at,
        n_threads=2
    )

    snowman1 = Snowman(np.array([-2., -1.2, 1.0]))
    snowman1.rotateY(math.radians(20))
    cube1 = Cube(Material(color=np.array([100., 100., 255.]), shininess=3.))
    cube1.translate(0., -1., 1.)
    cube1.scale(2., 1.5, 2.5)
    cube1.buildTriangles(camera)
    ramp1 = Ramp()
    ramp1.shearXY(0.5, True)
    ramp1.translate(-1., 0., 0.)
    ramp1.buildTriangles(camera)
    objects = [
        # Sphere(np.array([0., 1., 0.]), 0.4),
        # snowman1,
        # Plane(np.array([1., -1., 0.]), np.array([0., 1., 0.]), material=Material(shininess=5., texture=Texture('snow.jpg', 0.02))),
        Tree(np.array([0., -3., 6.])),
        # NatalTree(np.array([0., -1., 2.5])),
        # Portico(np.array([-3.5, -3., 6.]), camera)
        # Shed(np.array([-3.5, -1, 0.]), camera),
        # BVH(Sphere(cube1.center, cube1.radius), [cube1]),
        # ramp1,
        # Table(np.array([0.0, -1., 5.]), camera),
        # BasedCylinder(np.array([1.25, 0.95, 0.75]), np.array([0., 1., 0.]), 0.05, 1., center_top=np.array([1.25, 1., 0.75])),
    ]
    lights = [
        PointLight(np.array([1., 2.0, 5.0]), 0.6),
        DirectionalLight(np.array([-1., -0.2, 0.4]), 0.35),
        AmbientLight(0.35)
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
