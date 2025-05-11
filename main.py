import cv2
import numpy as np
from numba import cuda
import utils.camera as camera
import utils.window as window
from objects.light import Light
from utils.vector import rotateY
from objects.sphere import Sphere
from utils.world import ray_casting
from numba.cuda.random import create_xoroshiro128p_states


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
rng_states = create_xoroshiro128p_states(threads_per_block[0] * threads_per_block[1] * blocks_x * blocks_y, seed=1)

def render(tick):
    camera_position = rotateY(camera.position, np.pi / 180)
    camera.init(
        (width, height),
        camera_position,
        camera_at,
        new_viewport_size=(viewport_width, viewport_height),
    )
    device_camera = cuda.to_device(camera.get_device_camera())

    ray_casting[(blocks_x, blocks_y), threads_per_block](
        rng_states,
        device_image,
        device_camera,
        cuda.to_device(spheres),
        cuda.to_device(lights),
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
