from math import sqrt
from numba import cuda
from numpy import float64, inf
from utils.vector import RGB, Ray, v3, v3_div, v3_dot, v3_sub

type Sphere = tuple[v3, tuple[float64], RGB, tuple[float64]]

@cuda.jit(device=True)
def sphere_intersects(sphere: Sphere, ray: Ray):
    co = v3_sub(ray[0], sphere[0])

    b = 2. * v3_dot(co, ray[1])
    c = v3_dot(co, co) - (sphere[1][0] * sphere[1][0])
    delta = b * b - 4. * c

    if delta < 0.:
        return inf

    delta = sqrt(delta)
    t1 = (-b + delta) / 2
    t2 = (-b - delta) / 2

    t = inf
    if t1 > 0. and t1 < t:
        t = t1

    if t2 > 0. and t2 < t:
        t = t2

    return t - 0.0001


@cuda.jit(device=True)
def sphere_get_normal(sphere: Sphere, point: v3):
    return v3_div(v3_sub(point, sphere[0]), sphere[1][0])
