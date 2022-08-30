import cv2
import numpy as np
from utils.ray import Ray
from lights.lights import AmbientLight, Light
from utils.camera import Camera
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
        self.image = camera.buffer
        self.shadows = shadows
        self.camera.scene = self

    def update(self):
        if not self.loaded:
            self.camera.rayCast()
            self.loaded = True
            self.resize()

        glDrawPixels(self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE, self.image)

    def resize(self):
        if self.width != self.camera.resolution[0] or self.height != self.camera.resolution[1]:
            self.image = cv2.resize(self.camera.buffer, (self.width, self.height))

    def rayTrace(self, ray: Ray, target: Object = None, debug=False):
        point, target2, t = None, None, np.inf
        def __loop(object):
            if object.isComplex:
                for object in object.parts:
                    __loop(object)
                return

            nonlocal point, target2, t, ray, target
            if object is target:
                return

            aux = object.intersects(ray)
            if ray.t < t:
                target2 = object
                point = aux
                t = ray.t

        for object in self.objects:
            __loop(object)

        return point, target2

    def computeLightness(self, point: np.ndarray, normal: np.ndarray, target: Object):
        if self.shadows:
            lightness = np.array([0., 0., 0.])
            for light in self.lights:
                if light.ignoreShadow:
                    lightness += light.computeLight(point, normal) * light.color
                    continue

                lightDirection, lightDistance = light.getDirection(point)
                ray = Ray(point, lightDirection)
                ray.t = lightDistance
                _, target2 = self.rayTrace(ray, target=target)
                if ray.t >= lightDistance:
                    lightness += light.computeLight(point, normal) * light.color
        else:
            lightness = np.sum(light.computeLight(point, normal) * light.color for light in self.lights)
        return lightness
