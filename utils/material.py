import numpy as np
from numba import cuda
from abc import abstractmethod
from vector import Ray, v3, v3_mult, v3_reflect, v3_random_unit, v3_add, v3_is_near_zero

type CudaMaterial = tuple[v3, v3, v3]
n_rays_samples = 10

TYPE_METAL = 0
TYPE_LAMBERTIAN = 1

class Material:
    def __init__(self, color: np.ndarray, type: int, shininess: float = np.inf):
        self.color = color
        self.type = type
        self.shininess = shininess

    @abstractmethod
    def to_cuda(self) -> CudaMaterial:
        pass

    @abstractmethod
    def scatter(material: CudaMaterial):
        return material[0], 0

    @abstractmethod
    def get_scatters(material: CudaMaterial, ray: Ray, normal: v3, array: list[v3], n: int):
        pass


class Metal(Material):
    def __init__(self, color: np.ndarray, shininess: float, reflectance: float, roughness: float):
        super().__init__(color, TYPE_METAL, shininess)
        self.reflectance = reflectance
        self.roughness = roughness

    def to_cuda(self) -> CudaMaterial:
        return self.color, (self.type.value, self.shininess, self.reflectance), (self.roughness, 0., 0.)

    @cuda.jit(device=True)
    def scatter(material: CudaMaterial):
        return v3_mult(material[0], material[1][0]), 1

    @cuda.jit(device=True)
    def get_scatters(material: CudaMaterial, ray: Ray, normal: v3, array: list[v3], *args):
        array[0] = v3_reflect(ray[1], normal)

class Lambertian(Material):
    def __init__(self, color: np.ndarray, shininess: float = np.inf):
        super().__init__(color, TYPE_LAMBERTIAN, shininess)

    def to_cuda(self) -> CudaMaterial:
        return self.color, (self.type.value, self.shininess, 0.)

    @cuda.jit(device=True)
    def scatter(material: CudaMaterial):
        return material[0], n_rays_samples

    @cuda.jit(device=True)
    def get_scatters(material: CudaMaterial, ray: Ray, normal: v3, array: list[v3], n: int, rng_states, thread_id):
        for i in range(n):
            direction: v3 = v3_add(normal, v3_random_unit(rng_states, thread_id))
            if v3_is_near_zero(direction):
                direction = normal

            array[i] = direction

@cuda.jit(device=True)
def get_scatters(material: Material, *args):
    match material[1][0]:
        case TYPE_METAL.value:
            Metal.get_scatters(material, *args)
        case TYPE_LAMBERTIAN.value:
            Lambertian.get_scatters(material, *args)
