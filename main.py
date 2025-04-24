import math
import pygame
import numpy as np
from utils import transforms
from threading import Thread
from utils.material import Material, Texture
from utils.scene import Scene
from utils.window import Window
from utils.camera import Camera
from multiprocessing import cpu_count
from objects import Sphere, Plane
from lights.lights import AmbientLight, PointLight, DirectionalLight, SpotLight


def main():
    w, h = 400, 300
    camera_pos = np.array([0., 3., -6.0])
    camera_at = np.array([0., 1., 0.])
    camera = Camera(
        (w, h),
        camera_pos,
        camera_at,
        n_threads=cpu_count()-1,
        windowSize=np.array([600., 450.]),
        debounces=4,
    )

    spheres = [
        Sphere(
            np.array([0., 0.8, 1.]),
            0.8,
            Material(np.array([255., 50., 50.]), 1.0, reflectivity=0.5)
        ),
        Sphere(
            np.array([1.5, 0.8, -1.]),
            0.8,
            Material(np.array([50., 255., 50.]), 1.0, reflectivity=0.5)
        ),
        Sphere(
            np.array([-1.5, 0.8, -1.]),
            0.8,
            Material(np.array([50., 50., 255.]), 1.0, reflectivity=0.5)
        ),
    ]

    planes = [
        Plane(
            np.array([0., 0., 0.]),
            np.array([0., 1., 0.]),
            Material(None, 0.3, texture=Texture('empty_background.png', 0.2), reflectivity=0.5)
        ),
        Plane(
            np.array([.0, 0., 6.]),
            np.array([0., 0., -1.]),
            Material(np.array([127., 127., 127.]), 0.3, reflectivity=1., roughness=0.)
        ),
    ]

    lights = [
        PointLight(np.array([0., 3.0, 0.]), 0.7),
        # DirectionalLight(np.array([-1., -0.5, 1.0]), 0.5),
        AmbientLight(0.20),
    ]
    objects = [
        *spheres,
        *planes,
    ]
    scene = Scene(w, h, camera, objects, lights)
    window = Window(scene, title="Cube")
    window.open()
    window.startLoop()


if __name__ == "__main__":
    main()
