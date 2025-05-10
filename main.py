import cv2
import camera
import window
import numpy as np
from numba import cuda, njit
from light import Light, point_compute_lightness
from vector import rotateY, v3, v3_add, v3_clamp, v3_div, v3_mult_v3, v3_normalize, v3_sub, v3_mult, v3_madd
from sphere import Sphere, sphere_intersects, sphere_get_normal


@cuda.jit(device=True)
def get_sky_color(direction: v3):
    y = direction[1]
    blend = (y + 1) / 2

    blue = (0.5, 0.7, 1.0)
    white = (1., 1., 1.)

    return v3_madd(v3_mult(blue, blend), white, 1 - blend)

@cuda.jit
def render_image(image, camera: camera.Camera, spheres: tuple[Sphere], lights: tuple[Light]):
    x, y = cuda.grid(2)
    h, w = image.shape[:2]

    camera_position, pixel00_position, pixel_size, up, right = camera
    pixel_width, pixel_height, _ = pixel_size

    if x < w and y < h:
        n_spheres, n_lights = len(spheres), len(lights)
        pixel_center = v3_madd(pixel00_position, right, x * pixel_width)
        pixel_center = v3_madd(pixel_center, up, -y * pixel_height)
        ray_direction = v3_normalize(v3_sub(pixel_center, camera_position))

        min_t = np.inf
        min_sphere = spheres[0]
        for i in range(n_spheres):
            sphere = spheres[i]
            t = sphere_intersects(sphere, (camera_position, ray_direction))
            if t < min_t:
                min_t = t
                min_sphere = sphere

        if min_t != np.inf:
            hit_point = v3_madd(camera_position, ray_direction, min_t)
            normal = v3_normalize(sphere_get_normal(min_sphere, hit_point))
            # color = v3_mult_v3(v3_div(v3_add(normal, (1., 1., 1.)), 2), (1., 1., 1.))
            # color = (1., 0., 0.)
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
width, viewport_width = 1280, 720
height, viewport_height = int(width / aspect_ratio), int(viewport_width / aspect_ratio)

# camera definitions

camera_position = np.array([0., 0., -6.], dtype=np.float64)
camera_at = np.array([0., 0., 0.], dtype=np.float64)
camera.init(
    (width, height),
    camera_position,
    camera_at,
    new_viewport_size=(viewport_width, viewport_height),
)

device_camera = cuda.to_device(camera.get_device_camera())

threads_per_block = (16, 16)
blocks_x = (width + threads_per_block[0] - 1) // threads_per_block[0]
blocks_y = (height + threads_per_block[1] - 1) // threads_per_block[1]
device_image = cuda.to_device(np.zeros((height, width, 3), dtype=np.float64))

s0: Sphere = (
    (0., 0., 0.),
    (0.8, 0., 0.),
    (1., 0.1, 0.1),
    (1., 0., 0.),
)
spheres = (s0,)
l0: Light = (
    (0.5, 0.5, -2.),
    (0.7, 0., 0.),
    (1., 1., 1.),
)
lights = (l0,)

def render(tick):
    camera_position = rotateY(camera.position, np.pi / 180)
    camera.init(
        (width, height),
        camera_position,
        camera_at,
        new_viewport_size=(viewport_width, viewport_height),
    )
    device_camera = cuda.to_device(camera.get_device_camera())

    render_image[(blocks_x, blocks_y), threads_per_block](
        device_image,
        device_camera,
        cuda.to_device(spheres),  # spheres,
        cuda.to_device(lights),  # lights,
    )

    result_image = device_image.copy_to_host()

    return result_image

render_window = True
if render_window:
    window.init(width, height, render)
    window.open()
    window.startLoop()

final_image_bgr = render(0)[..., ::-1]
final_image = (final_image_bgr * 255.).astype(np.uint8)
cv2.imwrite("result.png", final_image)
