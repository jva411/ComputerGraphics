import numpy as np


class Material():
    def __init__(self, color = np.array([255., 255., 255.]), shininess = -1.):
        self.__color = color
        self.__shininess = shininess

    @property
    def color(self):
        return self.__color

    @property
    def shininess(self):
        return self.__shininess


BLANK = Material()
