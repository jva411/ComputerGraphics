import numpy as np
from utils.ray import Ray
from utils import transforms
from utils.material import Material


class Light:
    def __init__(self, position: np.ndarray, intensity: float, color: np.ndarray = np.array([255., 255., 255.])):
        self.position = position
        self.intensity = intensity
        self.color = color/255.
        self.on = True

    def computeLight(self, point: np.ndarray, normal: np.ndarray, ray: Ray, material: Material):
        return 0

    def getDirection(self, point: np.ndarray):
        direction = self.position - point
        distance = np.linalg.norm(direction)
        return direction / distance, distance

    @property
    def ignoreShadow(self):
        return False

class PointLight(Light):
    def __init__(self, position: np.ndarray, intensity, color: np.ndarray = np.array([255., 255., 255.])):
        super().__init__(position, intensity, color)

    def computeLight(self, point: np.ndarray, normal: np.ndarray, ray: Ray, material: Material):
        direction = self.position - point
        distance = np.linalg.norm(direction)
        direction = direction / distance

        dot = direction @ normal
        if dot <= 0: return 0

        sqrtD = (distance ** 0.5)
        lightness = self.intensity * dot / sqrtD
        if material.shininess == np.inf:
            return lightness

        r = 2*(direction @ normal) * normal - direction
        dot2 = r @ -ray.direction
        if dot2 > 0:
            lightness += self.intensity * (dot2 ** material.shininess) / sqrtD

        return lightness

class AmbientLight(Light):
    def __init__(self, intensity, color: np.ndarray = np.array([255., 255., 255.])):
        super().__init__(np.zeros(3), intensity, color)

    def computeLight(self):
        return self.intensity

    @property
    def ignoreShadow(self):
        return True


class DirectionalLight(Light):
    def __init__(self, direction: np.ndarray, intensity: np.ndarray, color = np.array([255., 255., 255.])):
        super().__init__(np.array([0., 0., 0.]), intensity, color)
        self.direction = transforms.normalize(-direction)

    def computeLight(self, point: np.ndarray, normal: np.ndarray, ray: Ray, material: Material):
        dot = self.direction @ normal
        if dot <= 0: return 0

        lightness = self.intensity * dot
        if material.shininess == np.inf:
            return lightness

        r = 2*(self.direction @ normal) * normal - self.direction
        dot2 = r @ -ray.direction
        if dot2 > 0:
            lightness += self.intensity * (dot2 ** material.shininess)

        return lightness

    def getDirection(self, point: np.ndarray):
        return self.direction, np.inf


class SpotLight(Light):
    def __init__(self, position: np.ndarray, direction: np.ndarray, intensity: float, angle: float, color = np.array([255., 255., 255.])):
        super().__init__(position, intensity, color)
        self.angle = angle
        self.cos = np.cos(self.angle)
        self.direction = transforms.normalize(direction)

    def computeLight(self, point: np.ndarray, normal: np.ndarray, ray: Ray, material: Material):
        direction = self.position - point
        distance = np.linalg.norm(direction)
        direction = direction / distance

        dot = direction @ normal
        if dot <= 0 or (direction @ self.direction < self.cos): return 0

        distance = np.linalg.norm(self.position - point)
        sqrtD = (distance ** 0.5)
        lightness = self.intensity * dot / sqrtD
        if material.shininess == np.inf:
            return lightness

        r = 2*(direction @ normal) * normal - direction
        dot2 = r @ -ray.direction
        if dot2 > 0:
            lightness += self.intensity * (dot2 ** material.shininess) / sqrtD

        return lightness
