import cv2
import numpy as np
from numba import cuda
from light import Light, point_compute_lightness
from vector import v3, v3_add, v3_clamp, v3_mult_v3, v3_normalize, v3_sub, v3_mult, v3_madd
from sphere import Sphere, sphere_intersects, sphere_get_normal
from window import Window


@cuda.jit(device=True)
def get_sky_color(direction: v3):
    y = direction[1]
    blend = (y + 1) / 2

    blue = (0.5, 0.7, 1.0)
    white = (1., 1., 1.)

    return v3_madd(v3_mult(blue, blend), white, 1 - blend)

@cuda.jit
def render_image(image, camera_origin, pixel00_position, pixel_width, pixel_height, spheres: tuple[Sphere], lights: tuple[Light]):
    x, y = cuda.grid(2)
    h, w = image.shape[:2]

    if x < w and y < h:
        n_spheres, n_lights = len(spheres), len(lights)
        pixel_center = (
            pixel00_position[0] + x * pixel_width,
            pixel00_position[1] - y * pixel_height,
            pixel00_position[2]
        )
        ray_direction = v3_normalize(v3_sub(pixel_center, camera_origin))

        min_t = np.inf
        min_sphere = spheres[0]
        for i in range(n_spheres):
            sphere = spheres[i]
            t = sphere_intersects(sphere, (camera_origin, ray_direction))
            if t < min_t:
                min_t = t
                min_sphere = sphere

        if min_t != np.inf:
            hit_point = v3_madd(camera_origin, ray_direction, t)
            normal = sphere_get_normal(min_sphere, hit_point)
            lightness = (0., 0., 0.)
            for i in range(n_lights):
                light = lights[i]
                new_lightness = point_compute_lightness(light, ray_direction, hit_point, normal, min_sphere[3][0])
                lightness = v3_add(lightness, new_lightness)

            color = v3_mult_v3(min_sphere[2], lightness)
        else:
            color = get_sky_color(ray_direction)

        color = v3_clamp(color)
        image[y, x, 0] = color[0]
        image[y, x, 1] = color[1]
        image[y, x, 2] = color[2]


# image size
aspect_ratio = 16. / 9.
width = 1280
height = int(width / aspect_ratio)

# camera definitions
camera_origin = np.array([0., 0., 0.], dtype=np.float64)
viewport_height = 2.
viewport_width = aspect_ratio * viewport_height

viewport_right = np.array([viewport_width, 0., 0.], dtype=np.float64)
viewport_down = np.array([0., -viewport_height, 0.], dtype=np.float64)
pixel_width = viewport_width / width
pixel_height = viewport_height / height
upper_left_corner = (camera_origin - np.array([0., 0., 1.])) - viewport_right / 2 - viewport_down / 2
pixel00_position = upper_left_corner + (pixel_width * np.array([1., 0., 0.]) + pixel_height * np.array([0., -1., 0.]))/2

device_camera_origin = cuda.to_device(camera_origin)
device_pixel00_position = cuda.to_device(pixel00_position)
device_image = cuda.to_device(np.zeros((height, width, 3), dtype=np.float64))

threads_per_block = (16, 16)
blocks_x = (width + threads_per_block[0] - 1) // threads_per_block[0]
blocks_y = (height + threads_per_block[1] - 1) // threads_per_block[1]

# TODO: fix camera orientation
s0: Sphere = (
    (0., 0., -1.),
    (0.5, 0., 0.),
    (1., 0.1, 0.1),
    (10., 0., 0.),
)
spheres = (s0,)
l0: Light = (
    (-.5, -.5, 2.),
    (0.7, 0., 0.),
    (1., 1., 1.),
)
lights = (l0,)

def render(tick):
    sphere: Sphere = (
        (0., 0., s0[0][2] + (tick % 100) * -0.1),
        s0[1],
        s0[2],
        s0[3],
    )
    spheres = (sphere,)
    cuda.to_device(spheres)

    render_image[(blocks_x, blocks_y), threads_per_block](
        device_image,
        device_camera_origin,
        device_pixel00_position,
        pixel_width,
        pixel_height,
        spheres,
        lights,
    )

    result_image = device_image.copy_to_host()

    return result_image

render_window = True
if render_window:
    window = Window(width, height, render)
    window.open()
    window.startLoop()

final_image_bgr = render(0)[..., ::-1]
final_image = (final_image_bgr * 255.).astype(np.uint8)
cv2.imwrite("result.png", final_image)
