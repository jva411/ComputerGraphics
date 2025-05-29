import os
import cv2
import numpy as np
from utils.ray import Ray
from utils import transforms


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

    def getColor(self, lightness: np.ndarray) -> np.ndarray:
        return self.__color * lightness

    def scatter(self, ray: Ray, point: np.ndarray, normal: np.ndarray) -> Ray:
        return None


class Metal(Material):
    def __init__(
        self,
        color=np.array([255, 255, 255]),
        shininess=np.inf,
        texture: Texture = None,
        reflectivity=0.,
        roughness=1.,
        fuzz=0.,
    ):
        super().__init__(color, shininess, texture)
        self.reflectivity = reflectivity
        self.roughness = roughness
        self.fuzz = fuzz

    def getColor(self, lightness):
        return super().getColor(lightness) * self.roughness

    def scatter(self, ray: Ray, point: np.ndarray, normal: np.ndarray) -> Ray:
        if self.reflectivity > 0:
            reflect_direction = transforms.reflect(ray.direction, normal)
            if self.fuzz > 0:
                reflect_direction += transforms.random_unit_vector() * self.fuzz
                reflect_direction = transforms.normalize(reflect_direction)

            reflect_ray = Ray(point, reflect_direction)
            return reflect_ray, 1.


class Lambertian(Material):
    def __init__(self, color=np.array([255, 255, 255]), shininess=np.inf, texture: Texture=None):
        super().__init__(color, shininess, texture)

    def scatter(self, ray: Ray, point: np.ndarray, normal: np.ndarray) -> Ray:
        diffuse_direction = normal + transforms.random_unit_vector()
        if (diffuse_direction < 1e-5).all():
            diffuse_direction = normal

        diffuse_direction = transforms.normalize(diffuse_direction)
        diffuse_ray = Ray(point, diffuse_direction)
        return diffuse_ray, 0.5


# class

BLANK = Material()
