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
from lights.lights import AmbientLight, PointLight, DirectionalLight
from objects.complex import Snowman, Tree, BasedCone, BasedCylinder, NatalTree


def main():
    w, h = 400, 300
    camera_pos = np.array([0., 2.0, -6.0])
    camera_at = np.array([0., 1., 0.])
    camera = Camera(
        (w, h),
        camera_pos,
        camera_at,
        n_threads=2
    )

    snowman1 = Snowman(np.array([-2., -1.2, 1.0]))
    snowman1.rotateY(math.radians(20))

    chair = Chair(np.array([-0.6, 1.1, -3.5]), camera)
    objects = [
        BVH(Sphere(chair.center.copy(), chair.radius), [chair])
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
