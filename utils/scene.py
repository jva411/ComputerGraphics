import cv2
import numpy as np
from utils.ray import Ray
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
        self.image = camera.buffer
        self.shadows = shadows
        self.camera.scene = self

    def update(self):
        if not self.loaded:
            self.camera.rayCast()
            self.image = self.camera.buffer.copy()
            self.loaded = True
            self.resize()

        glDrawPixels(self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE, self.image)

    def highlight(self, object: Object):
        self.image = self.camera.buffer.copy()
        for x, y in np.ndindex(*self.camera.resolution):
            obj = self.camera.pickingObjects[x, y]
            if obj is object: continue

            highlight = False
            for dx, dy in np.ndindex((2, 2)):
                if (
                    0 <= x + dx < self.camera.resolution[0] and
                    0 <= y + dy < self.camera.resolution[1] and (
                        self.camera.pickingObjects[x+dx, y+dy] is object or
                        self.camera.pickingObjects[x+dx, y-dy] is object or
                        self.camera.pickingObjects[x-dx, y+dy] is object or
                        self.camera.pickingObjects[x-dx, y-dy] is object
                    )
                ):
                    highlight = True
                    break

            if highlight:
                self.image[-y, x] = np.clip(self.image[-y, x] * np.array([2.5, 2.5, 1.2]), 0., 255.)

    def resize(self):
        if self.width != self.camera.resolution[0] or self.height != self.camera.resolution[1]:
            self.image = cv2.resize(self.camera.buffer, (self.width, self.height))

    def rayTrace(self, ray: Ray):
        point, target, t = None, None, np.inf
        def __loop(object):
            nonlocal ray
            if object.isComplex:
                if object.isBVH:
                    if object.intersects(ray) is not None:
                        for object in object.parts:
                            __loop(object)
                else:
                    for object in object.parts:
                        __loop(object)
                return

            nonlocal point, target, t
            aux = object.intersects(ray)
            if ray.t < t:
                target = object
                point = aux
                t = ray.t

        for object in self.objects:
            __loop(object)

        return point, target

    def computeLightness(self, point: np.ndarray, normal: np.ndarray, ray: Ray, target: Object):
        if self.shadows:
            lightness = np.array([0., 0., 0.])
            for light in self.lights:
                if light.ignoreShadow:
                    lightness += light.computeLight() * light.color
                    continue

                lightDirection, lightDistance = light.getDirection(point)
                ray2 = Ray(point, lightDirection)
                ray2.t = lightDistance
                self.rayTrace(ray2)
                if ray2.t >= lightDistance:
                    lightness += light.computeLight(point, normal, ray, target.material) * light.color
        else:
            lightness = np.sum(light.computeLight(point, normal, ray, target.material) * light.color for light in self.lights)
        return lightness
