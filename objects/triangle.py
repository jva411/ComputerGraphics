import numba
import numpy as np
from utils.ray import Ray
from utils import transforms
from utils.material import BLANK
from objects.plane import Plane, t_correction


class Triangle(Plane):
    def __init__(self, A: np.ndarray, B: np.ndarray, C: np.ndarray, material = BLANK):
        self.A = A
        self.B = B
        self.C = C
        self.AC = self.C - self.A
        self.AB = self.B - self.A
        self.BC = self.C - self.B
        self.area2 = np.linalg.norm(np.cross(self.AB, self.AC))
        self.area = self.area2 / 2
        self.normal = transforms.normalize(np.cross(self.AC, self.BC))
        super().__init__(self.A, self.normal, material)

    def intersects(self, ray: Ray) -> np.ndarray:
        return intersects(ray, self.position, self.normal, self.A, self.B, self.C, self.area2)


@numba.jit
def intersects(ray, position, normal, A, B, C, area2):
    dn = ray.direction @ normal
    if dn == 0: return None

    t = (position - ray.origin) @ normal / dn - t_correction
    if t < 0 or ray.t < t: return None

    p = ray.origin + ray.direction * t
    a1 = np.linalg.norm(np.cross(B-p, C-p))
    a2 = np.linalg.norm(np.cross(C-p, A-p))
    a3 = np.linalg.norm(np.cross(B-p, A-p))
    if np.abs(a1 + a2 + a3 - area2) > 0.00001:
        return None

    ray.t = t
    return p
