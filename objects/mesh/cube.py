import numpy as np
from objects.mesh import Mesh
from utils.material import BLANK


vertices = [
    np.array([0., 0., 0.]),
    np.array([0., 0., 1.]),
    np.array([1., 0., 1.]),
    np.array([1., 0., 0.]),
    np.array([0., 1., 0.]),
    np.array([0., 1., 1.]),
    np.array([1., 1., 1.]),
    np.array([1., 1., 0.])
]

edges = [
    (0, 1), # 0
    (0, 2), # 1
    (0, 3), # 2
    (0, 4), # 3
    (0, 5), # 4
    (1, 2), # 5
    (1, 5), # 6
    (1, 6), # 7
    (2, 3), # 8
    (2, 6), # 9
    (2, 7), # 10
    (3, 4), # 11
    (3, 7), # 12
    (4, 5), # 13
    (4, 6), # 14
    (4, 7), # 15
    (5, 6), # 16
    (6, 7)  # 17
]

faces = [  # Regra da mão direita
    ( 0,  6,  4),
    ( 4,  13, 3),
    ( 7, 16,  6),
    ( 5,  9,  7),
    (10, 17,  9),
    ( 8, 12, 10),
    ( 2,  3, 11),
    (11, 15, 12),
    ( 1,  5,  0),
    ( 2,  8,  1),
    (13, 16, 14),
    (14, 17, 15)
    # (14, 16, 13),
    # (15, 17, 14)
]


class Cube(Mesh):
    def __init__(self, material = BLANK):
        super().__init__(vertices, edges, faces, material)
