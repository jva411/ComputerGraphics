import math
import pygame
import numpy as np
from utils import transforms
from threading import Thread
from utils.material import Material, Lambertian, Metal
from utils.scene import Scene
from utils.window import Window
from utils.camera import Camera
from multiprocessing import cpu_count
from objects import Sphere, Plane
from lights.lights import AmbientLight, PointLight, DirectionalLight, SpotLight


def main():
    aspect_ratio = 16/9
    w_resolution, w_canvas = 400, 200
    resolution = (w_resolution, int(w_resolution / aspect_ratio))
    camera_pos = np.array([0., 1., -2.9])
    camera_at = np.array([0., 1., 0.])
    camera = Camera(
        resolution,
        camera_pos,
        camera_at,
        n_threads=cpu_count()-1,
        distance=0.8,
        windowSize=np.array([w_canvas, w_canvas / aspect_ratio], dtype=np.float64),
        debounces=5,
        n_samples=500,
        gamma_correction=True,
    )

    spheres = [
        # Sphere(
        #     np.array([0., 0.8, 1.]),
        #     0.8,
        #     Material(np.array([255., 50., 50.]), 1.0, reflectivity=0.5)
        # ),
        # Sphere(
        #     np.array([1.5, 0.8, -1.]),
        #     0.8,
        #     Material(np.array([50., 255., 50.]), 1.0, reflectivity=0.5)
        # ),
        # Sphere(
        #     np.array([-1.5, 0.8, -1.]),
        #     0.8,
        #     Material(np.array([50., 50., 255.]), 1.0, reflectivity=0.5)
        # ),
        Sphere(
            np.array([0., 0.8, 0.]),
            0.8,
            Lambertian(np.array([127., 127., 127.]), 1.0)
            # Metal(np.array([127., 127., 127.]), 1.0, reflectivity=0.7, roughness=0.)
        ),
        Sphere(
            np.array([-1.2, 0.5, -0.7]),
            0.5,
            Metal(np.array([127., 127., 127.]), 1.0, reflectivity=0.8, roughness=0.3)
        ),
        Sphere(
            np.array([1.2, 0.5, -0.7]),
            0.5,
            Metal(np.array([127., 127., 127.]), 1.0, reflectivity=0.8, roughness=0.3, fuzz=0.3)
        ),
    ]

    planes = [
        # Plane(
        #     np.array([0., 0., 0.]),
        #     np.array([0., 1., 0.]),
        #     Material(None, 10., texture=Texture('empty_background.png', 0.2), reflectivity=0.),
        # ),
        # Plane(
        #     np.array([.0, 0., 6.]),
        #     np.array([0., 0., -1.]),
        #     Material(np.array([127., 127., 127.]), 0.3, reflectivity=1., roughness=0.),
        # ),
        Plane(
            np.array([.0, 0., 0.]),
            np.array([0., 1., 0.]),
            Lambertian(np.array([255., 255., 255.]), 0.3),
        ),
        Plane(
            np.array([.0, 0., 2.]),
            np.array([0., 0., -1.]),
            Lambertian(np.array([255., 255., 255.]), 0.3),
        ),
        Plane(
            np.array([.0, 0., -3.]),
            np.array([0., 0., 1.]),
            Lambertian(np.array([255., 255., 255.]), 0.3),
        ),
        Plane(
            np.array([.0, 3., 0.]),
            np.array([0., -1., 0.]),
            Lambertian(np.array([255., 255., 255.]), 0.3),
        ),
        Plane(
            np.array([3.0, 0., 0.]),
            np.array([-1., 0., 0.]),
            Lambertian(np.array([255., 50., 50.]), 0.3),
        ),
        Plane(
            np.array([-3.0, 0., 0.]),
            np.array([1., 0., 0.]),
            Lambertian(np.array([50., 255., 50.]), 0.3),
        ),
    ]

    lights = [
        PointLight(np.array([0., 2.9, -2.]), 0.7),
        # DirectionalLight(np.array([-1., -0.5, 1.0]), 0.5),
        # AmbientLight(0.20),
    ]
    objects = [
        *spheres,
        *planes,
    ]
    scene = Scene(*resolution, camera, objects, lights)
    window = Window(scene, title="Cube")
    window.open()
    window.startLoop()


if __name__ == "__main__":
    main()
