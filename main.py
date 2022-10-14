import math
import pygame
import numpy as np
from objects.bvh import BVH
from utils import transforms
from utils.scene import Scene
from objects.mesh import Cube
from utils.window import Window
from utils.camera import Camera
from utils.material import Material
from objects.complex import Snowman, Tree
from objects import Sphere, Cone, Plane, Cylinder, Triangle
from lights.lights import AmbientLight, PointLight, DirectionalLight


def main():
    w, h = 400, 300
    camera_pos = np.array([-1., 2.0, -8.0])
    camera_at = np.array([0., 1.0, 0.])
    camera_up = np.array([0., 1., 0.])
    camera = Camera(
        (w, h),
        camera_pos,
        camera_at,
        camera_up
    )

    snowman1 = Snowman(np.array([0., -1.2, 2.0]))
    snowman1.rotate(math.radians(70), np.array([0., 1., 0.]))
    cube1 = Cube(Material(shininess=5.))
    cube1.buildTriangles(camera.direction)
    # snowman1.rotate(math.radians(90), np.array([0., 1., 0.]), np.array([0., 0., 0.]))
    objects = [
        # snowman1,
        # Plane(np.array([0., -1., 0.]), np.array([0., 1., 0.])),
        # Tree(np.array([-3., -1., 6.]))
        BVH(Sphere(np.array([0.5, 0.5, 0.5]), 0.8), [cube1])
    ]
    lights = [
        PointLight(np.array([1.5, 2.7, -2.]), 0.4),
        PointLight(np.array([-0.4, 1.0, -3.0]), 0.6),
        DirectionalLight(np.array([-1., -0.2, 0.2]), 0.2),
        AmbientLight(0.15)
    ]
    scene = Scene(w, h, camera, objects, lights)
    window = Window(scene, title="Cube")

    window.open()
    window.startLoop()


if __name__ == "__main__":
    main()
