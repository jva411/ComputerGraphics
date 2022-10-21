import math
import pygame
import numpy as np
from objects.bvh import BVH
from utils import transforms
from utils.scene import Scene
from objects.mesh import Cube
from utils.window import Window
from utils.camera import Camera
from utils.material import Material, Texture
from objects.complex import Snowman, Tree
from objects import Sphere, Cone, Plane, Cylinder, Triangle
from lights.lights import AmbientLight, PointLight, DirectionalLight


def main():
    w, h = 400, 300
    camera_pos = np.array([-15., 2.0, 4.0])
    camera_at = np.array([0., 1.0, 3.])
    camera = Camera(
        (w, h),
        camera_pos,
        camera_at
    )

    snowman1 = Snowman(np.array([-3., -1.2, 0.0]))
    snowman1.rotate(math.radians(70), np.array([0., 1., 0.]))
    cube1 = Cube(Material(color=np.array([100., 100., 255.]), shininess=3.))
    cube1.translate(-3.5, -1., 2.5)
    cube1.scale(2., 1.5, 2.5)
    cube1.buildTriangles(camera.direction)
    objects = [
        snowman1,
        Plane(np.array([0., -1., 0.]), np.array([0., 1., 0.]), material=Material(shininess=5., texture=Texture('snow.jpg', 0.02))),
        Tree(np.array([-3., -1., 6.])),
        BVH(Sphere(cube1.center, cube1.radius), [cube1])
    ]
    lights = [
        # PointLight(np.array([1.5, 2.7, -1.]), 0.4),
        PointLight(np.array([-5., 2.0, 4.0]), 0.6),
        DirectionalLight(np.array([1., -0.2, 0.4]), 0.35),
        AmbientLight(0.15)
    ]
    scene = Scene(w, h, camera, objects, lights)
    window = Window(scene, title="Cube")

    window.open()
    window.startLoop()


if __name__ == "__main__":
    main()
