import numpy as np


class Light:
    def __init__(self, position: np.ndarray, intensity: float, color: np.ndarray = np.array([255., 255., 255.])):
        self.position = position
        self.intensity = intensity
        self.color = color/255.

    def computeLight(self, point: np.ndarray, normal: np.ndarray):
        return 0

    @property
    def ignoreShadow(self):
        return False

class PointLight(Light):
    def __init__(self, position: np.ndarray, intensity, color: np.ndarray = np.array([255., 255., 255.])):
        super().__init__(position, intensity, color)

    def computeLight(self, point: np.ndarray, normal: np.ndarray):
        direction = self.position - point
        distance = np.linalg.norm(direction)
        direction = direction / distance

        dot = direction @ normal
        if dot <= 0: return 0

        return self.intensity * dot / (distance ** 0.5)

class AmbientLight(Light):
    def __init__(self, intensity, color: np.ndarray = np.array([255., 255., 255.])):
        super().__init__(np.zeros(3), intensity, color)

    def computeLight(self, point: np.ndarray, normal: np.ndarray):
        return self.intensity

    @property
    def ignoreShadow(self):
        return True
