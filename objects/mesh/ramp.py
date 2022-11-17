import numpy as np
from objects.mesh import Mesh
from utils.material import BLANK


vertices = [
    np.array([0., 0., 0.]),     # 0
    np.array([0., 0., 1.]),     # 1
    np.array([1., 0., 1.]),     # 2
    np.array([1., 0., 0.]),     # 3
    np.array([0., 1., 0.]),     # 4
    np.array([0., 1., 1.]),   # 5 -> 4
    # np.array([1., 1., 1.]),   # 6 -> 5
    # np.array([1., 1., 0.])      # 7 -> 5
]

edges = [
    (0, 1),     # 0
    (0, 2),     # 1
    (0, 3),     # 2
    (0, 4),     # 3
    # (0, 4),   # 4 -> 3
    (1, 2),     # 5 -> 4
    (1, 4),     # 6 -> 5
    (1, 5),     # 7 -> 6
    (2, 3),     # 8 -> 7
    (2, 5),     # 9 -> 8
    # (2, 5)    # 10 -> 8
    (3, 4),     # 11 -> 9
    (3, 5),     # 12 -> 10
    # (4, 4),   # 13 -> 10
    (4, 5),     # 14 -> 11
    # (4, 5),   # 15 -> 11
    # (5, 5),   # 16 -> 11
    # (5, 5)    # 17 -> 11
]

faces = [  # Regra da mão direita
    (0,  5,  3),
    # ( 3,  11, 3),
    (6, 11,  5),
    (4,  8,  6),
    # (8, 11,  8),
    (7, 10, 8),
    (2,  3, 9),
    (9, 11, 10),
    (1,  4,  0),
    (2,  7,  1),
    # (11, 11, 11),
    # (11, 11, 11)
]


class Ramp(Mesh):
    def __init__(self, material = BLANK):
        super().__init__([v.copy() for v in vertices], [(e[0], e[1]) for e in edges], [(f[0], f[1], f[2]) for f in faces], material)
