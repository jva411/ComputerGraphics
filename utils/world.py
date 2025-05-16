import numpy as np
from numba import cuda
import utils.camera as camera
from objects.light import Light, compute_lightness_point
from numba.cuda.random import xoroshiro128p_uniform_float64
from objects.sphere import Sphere, sphere_intersects, sphere_get_normal
from utils.vector import Ray, v3, v3_add, v3_clamp, v3_mult_v3, v3_normalize, v3_sub, v3_mult, v3_madd, v3_div


use_random_msaa = False
msaa_random, msaa_grid = 20, 4
msaa = msaa_random if use_random_msaa else msaa_grid
msaa_array_size = msaa if use_random_msaa else msaa * msaa
msaa_plus1 = msaa + 1


@cuda.jit(device=True)
def get_msaa_random_samples(rng_states, thread_id, samples):
    for sample in samples:
        sample[0] = xoroshiro128p_uniform_float64(rng_states, thread_id) - 0.5
        sample[1] = xoroshiro128p_uniform_float64(rng_states, thread_id) - 0.5

@cuda.jit(device=True)
def get_msaa_grid_samples(_, __, samples):
    for y in range(msaa):
        for x in range(msaa):
            samples[y * msaa + x][0] = x / msaa_plus1 - 0.5
            samples[y * msaa + x][1] = y / msaa_plus1 - 0.5

get_msaa_samples = get_msaa_random_samples if use_random_msaa else get_msaa_grid_samples

@cuda.jit(device=True)
def get_sky_color(y: np.float64):
    blend = (y + 1) / 2

    blue = (0.5, 0.7, 1.0)
    white = (1., 1., 1.)

    return v3_madd(v3_mult(blue, blend), white, 1 - blend)


@cuda.jit(device=True)
def send_ray(ray: Ray, spheres: tuple[Sphere]):
    n_spheres = len(spheres)
    min_t = np.inf
    min_sphere = spheres[0]
    for i in range(n_spheres):
        sphere = spheres[i]
        t = sphere_intersects(sphere, ray)
        if t < min_t:
            min_t = t
            min_sphere = sphere

    return min_t, min_sphere

@cuda.jit(device=True)
def compute_lightness(hit_point: v3, normal: v3, ray_direction: v3, shininess: np.float64, lights: tuple[Light]):
    n_lights = len(lights)
    lightness = (0., 0., 0.)
    for i in range(n_lights):
        light = lights[i]
        new_lightness = compute_lightness_point(light, ray_direction, hit_point, normal, shininess)
        lightness = v3_add(lightness, new_lightness)

    return lightness


@cuda.jit(device=True)
def ray_tracing(ray: Ray, spheres: tuple[Sphere], lights: tuple[Light]):
    min_t, min_sphere = send_ray(ray, spheres)
    if min_t == np.inf:
        return get_sky_color(ray[1][1])

    hit_point = v3_madd(*ray, min_t)
    normal = v3_normalize(sphere_get_normal(min_sphere, hit_point))

    # color = v3_mult_v3(v3_div(v3_add(normal, (1., 1., 1.)), 2), (1., 1., 1.))
    # color = (1., 0., 0.)
    lightness = compute_lightness(hit_point, normal, ray[1], min_sphere[3][0], lights)
    color = v3_mult_v3(min_sphere[2], lightness)

    return color

@cuda.jit
def ray_casting(rng_states, image, camera: camera.Camera, spheres: tuple[Sphere], lights: tuple[Light]):
    x, y = cuda.grid(2)
    h, w = image.shape[:2]
    thread_id = x + y * w

    camera_position, pixel00_position, pixel_size, up, right = camera
    pixel_width, pixel_height, _ = pixel_size

    if x < w and y < h:
        msaa_color = (0., 0., 0.)
        samples = cuda.local.array((msaa_array_size, 2), dtype=np.float64)
        get_msaa_samples(rng_states, thread_id, samples)
        for sample in samples:
            pixel_center = v3_madd(pixel00_position, right, (x + sample[0])* pixel_width)
            pixel_center = v3_madd(pixel_center, up, -(y + sample[1]) * pixel_height)
            ray_direction = v3_normalize(v3_sub(pixel_center, camera_position))
            ray = (camera_position, ray_direction)

            color = ray_tracing(ray, spheres, lights)

            color = v3_clamp(color)
            msaa_color = v3_add(msaa_color, color)

        color = v3_clamp(v3_div(msaa_color, msaa_array_size))
        image[y, x, 0] = color[0]
        image[y, x, 1] = color[1]
        image[y, x, 2] = color[2]
