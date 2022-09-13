import math
import pygame
import numpy as np
from utils import transforms
from utils.scene import Scene
from utils.window import Window
from utils.camera import Camera
from utils.material import Material
from objects.complex import Snowman
from objects import Sphere, Cone, Plane, Cylinder
from lights.lights import AmbientLight, PointLight, DirectionalLight


def main():
    snowman1 = Snowman(np.array([0., -1.2, 2.0]))
    snowman1.rotate(math.radians(70), np.array([0., 1., 0.]))
    # snowman1.rotate(math.radians(90), np.array([0., 1., 0.]), np.array([0., 0., 0.]))
    objects = [
        # Sphere(np.array([0, -0.5, 0]), .8),
        # Sphere(np.array([0, 0.6, 0]), .6),
        # Cone(np.array([0., 2., -2.]), np.array([0., 1., 1.]), 0.8, 0.4, material=Material(color=np.array([100., 100., 255.]), shininess=10)),
        # Sphere(np.array([0, 0.8, -2.6]), .2, material=Material(color=np.array([255, 0, 0]))),
        # Cylinder(np.array([0., 2., -1.]), np.array([0., 0.2, 1.]), 0.5, 0.4, material=Material(color=np.array([100., 100., 255.]))),
        snowman1,
        Plane(np.array([0., -1., 0.]), np.array([0., 1., 0.])),
    ]
    lights = [
        PointLight(np.array([1.5, 2.7, -2.]), 0.3),
        PointLight(np.array([-0.4, 1.0, -3.0]), 0.4),
        DirectionalLight(np.array([-1., -0.2, 0.2]), 0.08),
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
