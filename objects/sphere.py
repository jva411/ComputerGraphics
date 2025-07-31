import math
import numpy as np
from utils.ray import Ray
from utils import transforms
from utils.material import BLANK
from utils.core import dynamic_lib
from objects.object import Object, t_correction
from ctypes import CDLL, c_void_p, c_double


intersects = dynamic_lib.sphereIntersection
intersects.restype = c_double


class Sphere(Object):
    def __init__(self, position: np.ndarray, radius: float, material = BLANK):
        super().__init__(position, material)
        self.radius = radius
        self.positionP = None
        self.radiusC = None

    def preCalc(self, reverse=False):
        if reverse:
            self.positionP = None
            self.radiusC = None
        else:
            self.positionP = c_void_p(self.position.ctypes.data)
            self.radiusC = c_double(self.radius)

    def getNormal(self, point: np.ndarray) -> np.ndarray:
        geometric_normal = (point - self.position) / self.radius
        if self.material.texture is None or self.material.texture.normal_image is None:
            return geometric_normal

        p_rel = point - self.position
        theta = math.atan2(p_rel[2], p_rel[0])
        phi = math.acos(np.clip(p_rel[1] / self.radius, -1.0, 1.0))

        u = (theta + math.pi) / (2 * math.pi)
        v = 1.0 - (phi / math.pi)

        normal_from_map = self.material.texture.getNormal((u, v))
        if normal_from_map is None:
            return geometric_normal

        if abs(geometric_normal[1]) > 0.999:
            tangent = np.array([1., 0., 0.])
        else:
            tangent = transforms.normalize(np.cross(np.array([0., 1., 0.]), geometric_normal))

        bitangent = transforms.normalize(np.cross(geometric_normal, tangent))
        tbn = np.array([tangent, bitangent, geometric_normal]).T
        world_space_normal = tbn @ normal_from_map
        return transforms.normalize(world_space_normal)

    def getColor(self, point):
        if self.material.texture is None:
            return self.material.color

        p_rel = point - self.position
        theta = math.atan2(p_rel[2], p_rel[0])
        phi = math.acos(np.clip(p_rel[1] / self.radius, -1.0, 1.0))

        u = (theta + math.pi) / (2 * math.pi)
        v = 1.0 - (phi / math.pi)

        return self.material.texture.getColor((u, v))

    def intersects(self, ray: Ray) -> np.ndarray:
        t = intersects(ray.originP, ray.directionP, ray.tC, self.positionP, self.radiusC)
        if t>0:
            ray.t = t
            return ray.hitting_point
