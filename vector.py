import math
import numpy as np
from numba import cuda, jit

type v3 = tuple[np.float64, np.float64, np.float64]
type RGB = v3
type Ray = tuple[v3, v3]

@cuda.jit(device=True)
def v3_add(a: v3, b: v3):
    return a[0] + b[0], a[1] + b[1], a[2] + b[2]

@cuda.jit(device=True)
def v3_madd(a: v3, b: v3, s: np.float64):
    return a[0] + b[0] * s, a[1] + b[1] * s, a[2] + b[2] * s

@cuda.jit(device=True)
def v3_sub(a: v3, b: v3):
    return a[0] - b[0], a[1] - b[1], a[2] - b[2]

@cuda.jit(device=True)
def v3_mult(a: v3, s: np.float64):
    return a[0] * s, a[1] * s, a[2] * s

@cuda.jit(device=True)
def v3_mult_v3(a: v3, b: v3):
    return a[0] * b[0], a[1] * b[1], a[2] * b[2]

@cuda.jit(device=True)
def v3_div(a: v3, s: np.float64):
    return a[0] / s, a[1] / s, a[2] / s

@cuda.jit(device=True)
def v3_length(a: v3):
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])

@cuda.jit(device=True)
def v3_length2(a: v3):
    return a[0] * a[0] + a[1] * a[1] + a[2] * a[2]

@cuda.jit(device=True)
def v3_normalize(a: v3):
    length = math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])
    return a[0] / length, a[1] / length, a[2] / length

@cuda.jit(device=True)
def v3_dot(a: v3, b: v3):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

@cuda.jit(device=True)
def v3_cross(a: v3, b: v3):
    return a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]

@cuda.jit(device=True)
def v3_reflect(a: v3, normal: v3):
    return v3_sub(a, v3_mult(normal, 2 * v3_dot(a, normal)))

@cuda.jit(device=True)
def v3_clamp(a: v3):
    return max(min(a[0], 1.), 0.), max(min(a[1], 1.), 0.), max(min(a[2], 1.), 0.)


# CPU calcs

@jit
def normalize(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)

@jit
def rotate2D(v: np.ndarray, angle_radian: float):
    cos = math.cos(angle_radian)
    sin = math.sin(angle_radian)

    return np.array([
        cos*v[0] - sin*v[1],
        sin*v[0] + cos*v[1]
    ])

@jit
def rotate(vector: np.ndarray, angle_rad: float, axis: np.ndarray) -> np.ndarray:
    cos = math.cos(angle_rad / 2.0)
    b, c, d = -axis * math.sin(angle_rad / 2.0)
    aa, bb, cc, dd = cos * cos, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, cos * d, cos * c, cos * b, b * d, c * d

    M = np.array([
        [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
        [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
        [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]
    ])

    return M @ vector

@jit
def rotateX(vector: np.ndarray, angle_rad: float) -> np.ndarray:
    cos = math.cos(angle_rad)
    sin = math.sin(angle_rad)
    x, y, z = vector

    return np.array([
        x,
        y * cos - z * sin,
        y * sin + z * cos
    ])

@jit
def rotateY(vector: np.ndarray, angle_rad: float) -> np.ndarray:
    cos = math.cos(angle_rad)
    sin = math.sin(angle_rad)
    x, y, z = vector

    return np.array([
        x * cos + z * sin,
        y,
        -x * sin + z * cos
    ])

@jit
def rotateZ(vector: np.ndarray, angle_rad: float) -> np.ndarray:
    cos = math.cos(angle_rad)
    sin = math.sin(angle_rad)
    x, y, z = vector

    return np.array([
        x * cos - y * sin,
        x * sin + y * cos,
        z
    ])