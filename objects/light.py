from math import sqrt
from numba import cuda
from numpy import float64, inf
from utils.vector import RGB, v3, v3_div, v3_dot, v3_length, v3_mult, v3_sub

type Light = tuple[v3, tuple[float64], RGB]

@cuda.jit(device=True)
def compute_lightness_point(light: Light, ray_direction: v3, point: v3, normal: v3, shininess: float64):
    light_direction = v3_sub(light[0], point)
    distance = v3_length(light_direction)
    light_direction = v3_div(light_direction, distance)

    dot = v3_dot(light_direction, normal)
    if dot <= 0:
        return (0., 0., 0.)

    sqrt_distance = sqrt(distance)
    lightness = light[1][0] * dot / sqrt_distance
    if shininess == inf:
        return v3_mult(light[2], lightness)

    # TODO: fix highlight of specular light
    # reflected_direction = v3_reflect(light_direction, normal)
    pseudo_reflected_direction = v3_sub(v3_mult(normal, 2 * v3_dot(normal, light_direction)), light_direction)
    reflected_dot = v3_dot(pseudo_reflected_direction, v3_mult(ray_direction, -1))
    if reflected_dot > 0:
        lightness += light[1][0] * (reflected_dot ** shininess) / sqrt_distance

    return v3_mult(light[2], lightness)
