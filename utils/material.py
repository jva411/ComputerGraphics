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
        self.RGB = RGB
        if not RGB:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

    def getColor(self, point: np.ndarray) -> np.ndarray:
        x = int(point[0] / self.scale) % self.image.shape[1]
        y = int(point[1] / self.scale) % self.image.shape[0]
        return self.image[y, x]

    def copy(self):
        return Texture(self.path, self.scale, RGB=self.RGB)


class CubeMapTexture:
    def __init__(self, name: str):
        self.name = name
        self.faces = []
        cubemap_path = os.path.join(os.getcwd(), 'assets', 'textures', 'skyboxes', name)
        for i in range(1, 7):
            path = os.path.join(cubemap_path, f'{i}.bmp')
            if not os.path.exists(path):
                raise FileNotFoundError(f"Imagem do cubemap não encontrada em {path}")

            image = cv2.imread(path)
            self.faces.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    def getColor(self, direction: np.ndarray) -> np.ndarray:
        abs_dir = np.abs(direction)

        if abs_dir[0] > abs_dir[1] and abs_dir[0] > abs_dir[2]:
            face_index = 1 if direction[0] > 0 else 3 # Direita/Esquerda
            uc, vc = (-direction[2], -direction[1]) if direction[0] > 0 else (direction[2], -direction[1])
            sc = 0.5 / abs_dir[0]
        elif abs_dir[1] > abs_dir[2]:
            face_index = 4 if direction[1] > 0 else 5 # Cima/Baixo
            uc, vc = (direction[2], -direction[0]) if direction[1] > 0 else (direction[0], -direction[2])
            sc = 0.5 / abs_dir[1]
        else:
            face_index = 0 if direction[2] > 0 else 2 # Frente/Trás
            uc, vc = (direction[0], -direction[1]) if direction[2] > 0 else (-direction[0], -direction[1])
            sc = 0.5 / abs_dir[2]

        u = (uc * sc + 0.5)
        v = (vc * sc + 0.5)

        face_image = self.faces[face_index]
        height, width, _ = face_image.shape

        x = u * (width - 1)
        y = v * (height - 1)
        x0, y0 = int(x), int(y)
        x1, y1 = min(x0 + 1, width - 1), min(y0 + 1, height - 1)
        tx, ty = x - x0, y - y0

        c00 = face_image[y0, x0]
        c10 = face_image[y0, x1]
        c01 = face_image[y1, x0]
        c11 = face_image[y1, x1]

        c0 = c00 * (1 - tx) + c10 * tx
        c1 = c01 * (1 - tx) + c11 * tx
        return c0 * (1 - ty) + c1 * ty


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

    def getColor(self) -> np.ndarray:
        return self.__color

    def scatter(self, ray: Ray, point: np.ndarray, normal: np.ndarray) -> Ray:
        return None

    def copy(self):
        return Material(self.__color.copy(), self.__shininess, self.texture.copy() if self.texture else None)


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

    def getColor(self):
        return super().getColor() * self.roughness

    def scatter(self, ray: Ray, point: np.ndarray, normal: np.ndarray) -> Ray:
        if self.reflectivity > 0:
            reflect_direction = transforms.reflect(ray.direction, normal)
            if self.fuzz > 0:
                reflect_direction += transforms.random_unit_vector() * self.fuzz
                reflect_direction = transforms.normalize(reflect_direction)

            reflect_ray = Ray(point, reflect_direction)
            return reflect_ray, 1.

        return None, 0

    def copy(self):
        return Metal(
            self.color.copy(),
            self.shininess,
            self.texture.copy() if self.texture else None,
            self.reflectivity,
            self.roughness,
            self.fuzz
        )


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

    def copy(self):
        return Lambertian(
            self.color.copy(),
            self.shininess,
            self.texture.copy() if self.texture else None
        )


BLANK = Lambertian()
