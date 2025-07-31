import math
import pygame
import numpy as np
from utils import transforms
from threading import Thread
from utils.scene import Scene
from utils.window import Window
from utils.camera import Camera
from objects import Object, Plane
from objects.complex import Snowman
from multiprocessing import cpu_count
from utils.material import Material, Lambertian
from lights.lights import Light, AmbientLight, DirectionalLight


def main():
    aspect_ratio = 16/9
    w_resolution, w_canvas = 600, 200
    resolution = (w_resolution, int(w_resolution / aspect_ratio))
    camera_pos = np.array([0., 1.5, -1.])
    camera_at = np.array([0., 1.5, 0.])
    camera = Camera(
        resolution,
        camera_pos,
        camera_at,
        n_threads=cpu_count()-1,
        distance=0.6,
        windowSize=np.array([w_canvas, w_canvas / aspect_ratio], dtype=np.float64),
        debounces=0,
        n_samples=1,
        gamma_correction=True,
    )

    objects: list[Object] = []
    lights: list[Light] = []

    floor = Plane(
        np.array([0., 0., 0.]),
        np.array([0., 1., 0.]),
        Lambertian(np.array([255., 255., 255.]), 1.0)
    )
    objects.append(floor)

    snowman = Snowman(np.array([0., 0., 1.]), 0.5)
    objects.append(snowman)

    lights.append(AmbientLight(0.2, np.array([255., 255., 255.])))
    lights.append(DirectionalLight(np.array([-1., -0.2, 1.]), 0.6, np.array([255., 255., 255.])))

    scene = Scene(*resolution, camera, objects, lights)
    scene.load_cubemap("snowny_mountain")
    window = Window(scene, title="Cube")
    window.open()
    window.startLoop()


if __name__ == "__main__":
    main()
