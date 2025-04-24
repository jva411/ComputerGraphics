import os
import cv2
import numpy as np


class Texture():
    def __init__(self, path: str, scale = 1.0, RGB = True):
        self.path = os.path.join(os.getcwd(), 'assets', 'textures', path)
        self.scale = scale
        self.image = cv2.imread(self.path)
        if not RGB:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

    def getColor(self, point: np.ndarray) -> np.ndarray:
        x = int(point[0] / self.scale) % self.image.shape[1]
        y = int(point[1] / self.scale) % self.image.shape[0]
        return self.image[y, x]

class Material():
    def __init__(self, color = np.array([255., 255., 255.]), shininess = np.inf, texture: Texture = None, reflectivity = 0.0, roughness = 1.0):
        self.__color = color
        self.__shininess = shininess
        self.reflectivity = reflectivity
        self.roughness = roughness
        self.texture = texture

    @property
    def color(self):
        return self.__color

    @property
    def shininess(self):
        return self.__shininess


BLANK = Material()
