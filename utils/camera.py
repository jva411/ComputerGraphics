import math
import numpy as np
from utils.ray import Ray
from objects import Object
from utils import transforms
from lights.lights import Light


class Camera():
    def __init__(self, resolution: tuple[int, int], position: np.ndarray, at: np.ndarray, ratio = np.array([4, 3]), rotation=0):
        from utils.scene import Scene
        self.scene: Scene = None

        self.position = position
        self.direction = transforms.normalize(at - position)
        self.resolution = np.array([*resolution], dtype=np.uint32)
        self.ratio = ratio
        self.buffer = np.zeros((*resolution[::-1], 3), dtype=np.float64)
        self.pickingObjects = np.empty(resolution, dtype=Object)
        # print(self.buffer.shape, self.pickingObjects)

        dirXZ = self.direction[[0, 2]]
        if all(dirXZ == np.array([0., 0.])):
            aXZ = 0
        else:
            rDirXZ = transforms.rotate2D(transforms.normalize(dirXZ), -np.pi/2)
            dX = rDirXZ @ np.array([1, 0])
            dZ = rDirXZ @ np.array([0, 1])
            aXZ = np.arccos(dX)
            if (dZ < 0): aXZ = 2*np.pi - aXZ

        self.right = transforms.rotateY(np.array([1., 0., 0.]), -aXZ)
        self.up = transforms.rotate(self.direction, -np.pi/2, self.right)
        if rotation > 0:
            self.up = transforms.rotate(self.up, np.radians(rotation), self.direction)
            self.right = transforms.rotate(self.right, np.radians(rotation), self.direction)

    def rayCast(self):
        for x, y in np.ndindex(*self.resolution):
            ray = Ray(self.position, self.get_ray_direction(x, y))
            point, target = self.scene.rayTrace(ray)

            if target is None:
                self.buffer[y, x] = [203/10, 224/10, 233/10]
            else:
                normal = target.getNormal(point)
                lightness = self.scene.computeLightness(point, normal, ray, target)
                self.buffer[y, x] = np.clip(target.getColor(point) * lightness, 0., 255.)

                self.pickingObjects[x, -y] = target

    def get_ray_direction(self, x: int, y: int) -> np.ndarray:
        frameO = self.position + self.direction*5
        dx, dy = ((np.array([x, y]) - self.resolution/2) / self.resolution) * self.ratio
        pixelPos = frameO + dy*self.up + dx*self.right
        direction = pixelPos - self.position
        return direction/np.linalg.norm(direction)
