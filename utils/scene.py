import cv2
import time
import numpy as np
from utils.ray import Ray
from threading import Thread
from utils import transforms
from utils.camera import Camera
from lights.lights import Light
from objects import Object, Cone
from OpenGL.GL import glDrawPixels, GL_RGB, GL_UNSIGNED_BYTE


class Scene:
    def __init__(self, width: int, height: int, camera: Camera, objects: list[Object], lights: list[Light], shadows=True):
        self.width = width
        self.height = height
        self.camera = camera
        self.objects = objects
        self.lights = lights
        self.loaded = False
        self.loading = False
        self.image = camera.buffer
        self.shadows = shadows
        self.camera.scene = self

    def __threadedRaycast(self):
            t0 = time.time()
            self.camera.rayCast(self)
            print(time.time() - t0)
            self.loading = False
            self.loaded = True


    def update(self):
        if not self.loaded and not self.loading:
            self.loading = True
            Thread(target=self.__threadedRaycast, daemon=True).start()

        glDrawPixels(self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE, self.image)

    def rayTrace(self, ray: Ray):
        point, target, t = None, None, np.inf
        def __loop(object, ray, simulate=False, test=False):
            if object.isComplex:
                if object.isBVH:
                    if (
                        object.bounding.isComplex and any(__loop(object, Ray(ray.origin, ray.direction, ray.t), True) for object in object.bounding.parts)
                        or object.intersects(ray) is not None
                    ):
                        for object in object.parts:
                            __loop(object, ray)
                else:
                    for object in object.parts:
                        __loop(object, ray)
                return

            nonlocal point, target, t
            aux = object.intersects(ray)
            if ray.t < t:
                if simulate: return True

                target = object
                point = aux
                t = ray.t

        for object in self.objects:
            __loop(object, ray)

        return point, target

    def computeLightness(self, point: np.ndarray, normal: np.ndarray, ray: Ray, target: Object):
        if self.shadows:
            lightness = np.array([0., 0., 0.])
            for light in self.lights:
                if light.ignoreShadow:
                    lightness += light.computeLight() * light.color
                    continue

                lightDirection, lightDistance = light.getDirection(point)
                ray2 = Ray(point, transforms.normalize(lightDirection))
                ray2.t = lightDistance
                self.rayTrace(ray2)
                if ray2.t >= lightDistance:
                    lightness += light.computeLight(point, normal, ray, target.material) * light.color
        else:
            lightness = np.sum(light.computeLight(point, normal, ray, target.material) * light.color for light in self.lights)
        return lightness
