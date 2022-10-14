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
        dn = ray.direction @ self.normal
        if dn == 0: return None

        t = (self.position - ray.origin) @ self.normal / dn - t_correction
        if t < 0 or ray.t < t: return None

        p = ray.origin + ray.direction * t
        a1 = np.linalg.norm(np.cross(self.B-p, self.C-p))
        a2 = np.linalg.norm(np.cross(self.C-p, self.A-p))
        a3 = np.linalg.norm(np.cross(self.B-p, self.A-p))
        if np.abs(a1 + a2 + a3 - self.area2) > 0.00001:
            return None

        ray.t = t
        return ray.hitting_point
