import math
import numpy as np


def normalize(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)

def rotateX(vector: np.ndarray, angle: float) -> np.ndarray:
    cos = math.cos(angle)
    sin = math.sin(angle)
    x, y, z = vector
    return np.array([
        x,
        y * cos - z * sin,
        y * sin + z * cos
    ])

def rotateY(vector: np.ndarray, angle: float) -> np.ndarray:
    cos = math.cos(angle)
    sin = math.sin(angle)
    x, y, z = vector
    return np.array([
        x * cos + z * sin,
        y,
        -x * sin + z * cos
    ])

def rotateZ(vector: np.ndarray, angle: float) -> np.ndarray:
    cos = math.cos(angle)
    sin = math.sin(angle)
    x, y, z = vector
    return np.array([
        x * cos - y * sin,
        x * sin + y * cos,
        z
    ])

def rotate(vector: np.ndarray, angle: float, axis: np.ndarray) -> np.ndarray:
    cos = math.cos(angle / 2.0)
    b, c, d = -axis * math.sin(angle / 2.0)
    aa, bb, cc, dd = cos * cos, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, cos * d, cos * c, cos * b, b * d, c * d

    M = np.array([
        [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
        [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
        [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]
    ])

    return M @ vector


def rotate2D(v, angle):
    cos = math.cos(angle)
    sin = math.sin(angle)
    return np.array([
        cos*v[0] - sin*v[1],
        sin*v[0] + cos*v[1]
    ])
