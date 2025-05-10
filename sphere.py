from math import sqrt
from numba import cuda
from numpy import float64, inf
from vector import v3, v3_add, v3_dot, v3_normalize, v3_sub, RGB, Ray

type Sphere = tuple[v3, tuple[float64], RGB, tuple[float64]]

@cuda.jit(device=True)
def sphere_intersects(sphere: Sphere, ray: Ray):
    oc = v3_sub(ray[0], sphere[0])

    a = v3_dot(ray[1], ray[1])
    b = -2. * v3_dot(oc, ray[1])
    c = v3_dot(oc, oc) - (sphere[1][0] * sphere[1][0])

    delta = b * b - 4 * a * c

    if delta < 0.:
        return inf

    t1 = (-b - sqrt(delta)) / (2 * a)
    t2 = (-b + sqrt(delta)) / (2 * a)

    t = inf
    if t1 > 0.:
        t = t1
    if t2 < t:
        t = t2

    return t - 0.0001


@cuda.jit(device=True)
def sphere_get_normal(sphere: Sphere, point: v3):
    return v3_normalize(v3_sub(point, sphere[0]))
