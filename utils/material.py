import numpy as np


class Texture():
    def __init__(self, path: str, scale = 1.0):
        self.path = path
        self.scale = scale


class Material():
    def __init__(self, color = np.array([255., 255., 255.]), shininess = np.inf, texture: Texture = None):
        self.__color = color
        self.__shininess = shininess
        self.texture = texture

    @property
    def color(self):
        return self.__color

    @property
    def shininess(self):
        return self.__shininess


BLANK = Material()
