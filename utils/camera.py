import numpy as np
from utils.ray import Ray
from objects import Object
from utils import transforms
from lights.lights import Light


class Camera():
    def __init__(self, resolution: tuple[int, int], position: np.ndarray, at: np.ndarray, up: np.ndarray, ratio = np.array([4, 3])):
        from utils.scene import Scene
        self.scene: Scene = None

        self.position = position
        self.direction = transforms.normalize(at - position)
        self.resolution = np.array([*resolution], dtype=np.uint32)
        self.ratio = ratio
        self.buffer = np.zeros((*resolution[::-1], 3), dtype=np.float64)

        pos_at = position - at
        distance = np.linalg.norm(pos_at[[0, 2]])
        self.up = transforms.rotate(
            transforms.normalize(up),
            np.arctan(pos_at[1]/distance),
            axis=np.array([1., 0., 0.])
        )
        self.right = transforms.rotate(self.up, np.pi/2, self.direction)

    def rayCast(self):
        for x, y in np.ndindex(*self.resolution):
            ray = Ray(self.position, self.get_ray_direction(x, y))
            point, target = self.scene.rayTrace(ray)

            if target is None:
                self.buffer[y, x] = [203, 224, 233]
            else:
                normal = target.getNormal(point)
                lightness = self.scene.computeLightness(point, normal, ray, target)
                self.buffer[y, x] = np.clip(target.color * lightness, 0., 255.)

    def get_ray_direction(self, x: int, y: int) -> np.ndarray:
        frameO = self.position + self.direction*5
        dx, dy = ((np.array([x, y]) - self.resolution/2) / self.resolution) * self.ratio
        pixelPos = frameO + dy*self.up + dx*self.right
        direction = pixelPos - self.position
        return direction/np.linalg.norm(direction)
