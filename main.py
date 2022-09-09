import math
import pygame
import numpy as np
from utils import transforms
from utils.scene import Scene
from utils.window import Window
from utils.camera import Camera
from lights.lights import AmbientLight, PointLight, DirectionalLight
from objects import Sphere, Cone, Plane, Cylinder
from objects.complex import Snowman


def main():
    snowman1 = Snowman(np.array([0., -1.2, 2.0]))
    snowman1.rotate(math.radians(87), np.array([0., 1., 0.]))
    # snowman1.rotate(math.radians(90), np.array([0., 1., 0.]), np.array([0., 0., 0.]))
    objects = [
        # Sphere(np.array([0, -0.5, 0]), .8),
        # Sphere(np.array([0, 0.6, 0]), .6),
        # Cone(np.array([0., 2., -2.]), np.array([0., 1., 1.]), 0.8, 0.4, color=np.array([100., 100., 255.])),
        # Sphere(np.array([0, 0.8, -2.6]), .2, color=np.array([255, 0, 0])),
        # Cylinder(np.array([0., 2., -1.]), np.array([0., 0.2, 1.]), 0.5, 0.4, color=np.array([100., 100., 255.])),
        snowman1,
        Plane(np.array([0., -1., 0.]), np.array([0., 1., 0.])),
    ]
    lights = [
        PointLight(np.array([1., 1.8, -4.8]), 0.5),
        PointLight(np.array([0., 1.0, -3.0]), 0.6),
        DirectionalLight(np.array([-1., -0.2, 0.2]), 0.26),
        AmbientLight(0.08)
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
