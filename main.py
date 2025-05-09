import cv2
import numpy as np
from numba import cuda

@cuda.jit(device=True)
def get_sky_color(direction: tuple[float, float, float]):
    y = direction[1]
    blend = (y + 1) / 2

    blue = (0.5, 0.7, 1.0)
    white = (1., 1., 1.)

    r = blue[0] * blend + white[0] * (1 - blend)
    g = blue[1] * blend + white[1] * (1 - blend)
    b = blue[2] * blend + white[2] * (1 - blend)

    return r, g, b

@cuda.jit
def render_image(image):
    y, x = cuda.grid(2)
    w, h = image.shape[:2]

    if x < w and y < h:
        ray_y = (h/2 - x) / (h/2)
        color = get_sky_color((0., ray_y*2, 0.))

        for channel in range(3):
            image[x, y, channel] = color[channel]


# image size
aspect_ratio = 16. / 9.
width = 1920
height = int(width / aspect_ratio)

# camera definitions
origin = np.array([0., 0., 0.], dtype=np.float64)
viewport_height = 2.
viewport_width = aspect_ratio * viewport_height

horizontal = np.array([viewport_width, 0., 0.], dtype=np.float64)
vertical = np.array([0., -viewport_height, 0.], dtype=np.float64)
pixel_size_x = horizontal / width
pixel_size_y = vertical / height
upper_left_corner = (origin - np.array([0., 0., 1.])) - horizontal / 2 - vertical / 2

# wait

image = cuda.to_device(np.zeros((height, width, 3), dtype=np.float64))

threads_per_block = (16, 16)
blocks_x = (width + threads_per_block[0] - 1) // threads_per_block[0]
blocks_y = (height + threads_per_block[1] - 1) // threads_per_block[1]


render_image[(blocks_x, blocks_y), threads_per_block](image)

result_image = image.copy_to_host()
result_image_bgr = result_image[..., ::-1]
final_image = (result_image_bgr * 255.).astype(np.uint8)
cv2.imwrite("result.png", final_image)
