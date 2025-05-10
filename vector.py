import numpy as np
from math import sqrt
from numba import cuda

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
    return sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])

@cuda.jit(device=True)
def v3_length2(a: v3):
    return a[0] * a[0] + a[1] * a[1] + a[2] * a[2]

@cuda.jit(device=True)
def v3_normalize(a: v3):
    length = sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])
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
