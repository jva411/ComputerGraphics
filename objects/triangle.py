import numpy as np
from utils.ray import Ray
from utils import transforms
from utils.material import BLANK
from utils.core import dynamic_lib
from objects.plane import Plane, t_correction
from ctypes import CDLL, c_void_p, c_double


intersects = dynamic_lib.triangleIntersection
intersects.restype = c_double


class Triangle(Plane):
    def __init__(self, A: np.ndarray, B: np.ndarray, C: np.ndarray, material = BLANK):
        self.A = A
        self.B = B
        self.C = C
        self.normal = transforms.normalize(np.cross(A-C, B-C))
        self.area = np.cross(self.A-B, self.A-C) @ self.normal
        super().__init__(self.A, self.normal, material)

        self.AP = None
        self.BP = None
        self.CP = None
        self.positionP = None
        self.normalP = None
        self.areaC = None

    def preCalc(self, reverse=False):
        if reverse:
            self.AP = None
            self.BP = None
            self.CP = None
            self.positionP = None
            self.normalP = None
            self.areaC = None
        else:
            self.AP = c_void_p(self.A.ctypes.data)
            self.BP = c_void_p(self.B.ctypes.data)
            self.CP = c_void_p(self.C.ctypes.data)
            self.positionP = c_void_p(self.position.ctypes.data)
            self.normalP = c_void_p(self.normal.ctypes.data)
            self.areaC = c_double(self.area)


    def intersects(self, ray: Ray) -> np.ndarray:
        t = intersects(ray.originP, ray.directionP, ray.tC, self.positionP, self.normalP, self.AP, self.BP, self.CP, self.areaC)
        if t>0:
            ray.t = t
            return ray.hitting_point
