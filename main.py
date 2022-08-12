import math
import pygame
import numpy as np
from utils import transforms
from utils.scene import Scene
from utils.window import Window
from objects.camera import Camera
from objects.lights import AmbientLight, PointLight
from objects.gameobjects import Sphere, Cone, Snowman, Plane, BasedCone


def main():
    snowman1 = Snowman(np.array([0., -1.2, 2.0]))
    snowman1.rotate(math.radians(87), np.array([0., 1., 0.]))
    # snowman1.rotate(math.radians(90), np.array([0., 1., 0.]), np.array([0., 0., 0.]))
    objects = [
        # Sphere(np.array([0, -0.5, 0]), .8),
        # Sphere(np.array([0, 0.6, 0]), .6),
        # BasedCone(np.array([0., 2., -2.]), np.array([0., 1., 1.]), 0.8, 0.4, color=np.array([100., 100., 255.])),
        Cone(np.array([0., 2., -2.]), np.array([0., 1., 1.]), 0.8, 0.4, color=np.array([100., 100., 255.])),
        # Circle(np.array([0., 1.2, -2.8]), np.array([0., -1., -1.]), 0.4, color=np.array([100., 100., 255.])),
        # Sphere(np.array([0, 0.8, -2.6]), .2, color=np.array([255, 0, 0])),
        snowman1,
        Plane(np.array([0., -1., 0.]), np.array([0., 1., 0.])),
    ]
    lights = [
        # PointLight(np.array([1., 1.8, -4.8]), 0.8),
        PointLight(np.array([0., 1.2, -2.8]), 0.9),
        AmbientLight(0.12)
    ]

    w, h = 400, 300
    camera_pos = np.array([0, 3.0, -8.0])
    camera_at = np.array([0., 1., 0.])
    camera_up = np.array([0., 1., 0.])
    camera = Camera(
        (w, h),
        camera_pos,
        camera_at,
        camera_up
    )
    scene = Scene(w, h, camera, objects, lights)
    window = Window(scene, title="Cube")

    window.open()
    window.startLoop()


if __name__ == "__main__":
    main()
